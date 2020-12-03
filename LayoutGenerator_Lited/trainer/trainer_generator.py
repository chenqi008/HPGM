import os
import math
import time
import json
import torch
import random
import cv2 as cv
import numpy as np
import torch.optim as optim
from miscc.config import cfg
from miscc.utils import mkdir_p
import torch.backends.cudnn as cudnn
from model.model_LSTM import LayoutEvaluator
from model.model_LSTM import LayoutGenerator
import matplotlib.pyplot as plt
from miscc.getContour import draw_point
from trainer.trainer_evaluator import visualization, tran_score_range
from miscc.getContour import ConversionLayout
plt.switch_backend('agg')


def define_optimizers(model, lr, weight_decay):
    optimizer_model = optim.Adam(model.parameters(), lr=lr,
                                 weight_decay=weight_decay, betas=(0.5, 0.999))
    # optimizer_model = optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay, alpha=0.99, eps=1e-08,
    #                                 momentum=0, centered=False)
    # optimizer_model = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    return optimizer_model

def save_model(model, epoch, model_dir, model_name, best=False):
    if best:
        torch.save(model.state_dict(), '%s/%s_best.pth' % (model_dir, model_name))
    else:
        torch.save(model.state_dict(), '%s/%s_%d.pth' % (model_dir, model_name, epoch))


class evaluateMetric(object):
    def __init__(self):
        pass

    def get_line_from_room(self, room_hull):
        """
        function: get the line from room_hull
        """
        line_set = []
        for h in range(len(room_hull)):
            if h + 1 == len(room_hull):
                line_set.append([room_hull[h], room_hull[0]])
            else:
                line_set.append([room_hull[h], room_hull[h + 1]])
        return line_set

    def calculate_distance(self, point1, point2):
        """
        function: calculate the distance between two points
        """
        [x1, y1] = np.array(point1, dtype=np.float)
        [x2, y2] = np.array(point2, dtype=np.float)
        distance = np.power((y2 - y1), 2) + np.power((x2 - x1), 2)
        distance = np.sqrt(distance)
        return distance

    def get_distance_from_room(self, hull):
        line_set = self.get_line_from_room(hull)
        hull_distances = []
        for s in range(len(line_set)):
            distance = self.calculate_distance(line_set[s][0], line_set[s][1])
            hull_distances.append(distance)
        return hull_distances

    @staticmethod
    def line_is_on_contour(line, point_hull):
        """
        function: judge the line on contour or not
        """
        line[0], line[1] = np.array(line[0]), np.array(line[1])
        valid_line = (line[0] - line[1]).astype(int)
        for p in range(len(point_hull)):
            start_point = point_hull[p]
            if p + 1 == len(point_hull):
                end_point = point_hull[0]
            else:
                end_point = point_hull[p + 1]
            contour_line = (end_point - start_point).astype(int)
            on_line = (start_point - line[0]).astype(int)
            # on line or on extension line; cv.pointPolygonTest: +1(inside); -1(outside); 0(on the edge)
            if np.cross(valid_line, contour_line) == 0 and np.cross(valid_line, on_line) == 0:
                if cv.pointPolygonTest(point_hull, (line[0][0], line[0][1]), False) == 0 \
                        and cv.pointPolygonTest(point_hull, (line[1][0], line[1][1]), False) == 0:
                    return True
        return False

    @staticmethod
    def calculate_surface_area(polygon):
        # use the image to calculate the area
        im = np.zeros((256, 256))
        polygon_mask = cv.fillPoly(im, [np.array(polygon, dtype=np.int)], 255)
        area = np.sum(np.greater(polygon_mask, 0))
        return area

    def calculate_score_ratio(self, room_hull):
        """
        function: calculate the score_ratio of the contour hull
        """
        distances = self.get_distance_from_room(room_hull)
        sorted_distances = sorted(distances)
        max_distance, min_distance = sorted_distances[-1], sorted_distances[0]
        ratio = float(min_distance) / float(max_distance)
        p_max, p_min, score_ratio = 3., 1./3., 0
        if ratio > p_max:
            score_ratio = ratio / p_max
        elif p_min <= ratio and ratio <= p_max:
            score_ratio = 0
        else:
            score_ratio = p_min / ratio
        return score_ratio

    def calculate_score_area(self, room_num, room_hull, init_contour):
        """
        function:calculate the score_area of the room
        """
        k_min = 1. / (2* room_num)
        area = float(self.calculate_surface_area(room_hull))
        total_area = float(self.calculate_surface_area(init_contour))
        k = area / total_area
        if k <= k_min:
            score_area = k_min / k
        else:
            score_area = 0
        return score_area

    def calculate_score_cost(self, room_hull, contour):
        """
        function:calculate the score_cost of the room
        """
        contour = np.array(contour)
        score_cost = 0.0
        line_set = self.get_line_from_room(room_hull)
        for m in range(len(line_set)):
            if not self.line_is_on_contour(line_set[m], np.array(contour)):
                score_cost = self.calculate_distance(line_set[m][0], line_set[m][1])
        return score_cost

class LayGenerator(object):
    def __init__(self, output_dir, dataloader_train, dataloader_test, logger=None):
        self.output_dir = output_dir
        self.model_dir = os.path.join(output_dir, 'Model')
        self.log_dir = os.path.join(output_dir, 'Log')
        self.eval_dir = os.path.join(output_dir, 'eval')
        self.eval_model_path = cfg.EVAL.MODEL_EVALUATOR
        self.gen_model_path = cfg.EVAL.MODEL_GENERATOR
        self.logger = logger
        mkdir_p(self.model_dir)
        mkdir_p(self.log_dir)
        mkdir_p(self.eval_dir)
        # save the information of cfg
        log_cfg = os.path.join(self.log_dir, 'log_cfg.json')
        with open(log_cfg, 'a') as outfile:
            json.dump(cfg, outfile)
            outfile.write('\n')
        cudnn.benchmark = True
        self.max_epoch = cfg.TRAIN.MAX_EPOCH
        if cfg.TRAIN.FLAG:
            self.dataloader_train = dataloader_train
        self.dataloader_test = dataloader_test
        self.best_loss = 10000.0
        self.best_epoch = 0

    def define_models(self):
        def weight_init(m):
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.orthogonal(m.weight)
                # torch.nn.init.constant_(m.bias, 0.01)
        room_dim = 4
        max_len = 50
        room_hiddern_dim = cfg.TRAIN.ROOM_HIDDERN_DIM
        score_hiddern_dim = cfg.TRAIN.SCORE_HIDDERN_DIM
        gen_room_hiddern_dim = cfg.TRAIN.ROOM_GEN_HIDDERN_DIM
        init_hidden_dim = [3, 1]
        layout_evaluator = LayoutEvaluator(4, 8, score_hiddern_dim, bidirectional=False)
        layout_generator = LayoutGenerator(room_dim, room_hiddern_dim,
                                           gen_room_hiddern_dim, init_hidden_dim, max_len, self.logger, bidirectional=False)
        layout_generator.apply(weight_init)
        # for param in layout_generator.parameters():
        #     param.data.uniform_(-0.08, 0.08)
        return layout_evaluator, layout_generator

    def train(self):
        layout_evaluator, layout_generator = self.define_models()
        self.logger.info("Layout_evaluator: {}".format(layout_evaluator))
        self.logger.info("Layout_generator: {}".format(layout_generator))
        if cfg.TRAIN.EVALUATOR != '':
            layout_evaluator.load_state_dict(torch.load(cfg.TRAIN.EVALUATOR))
        if cfg.TRAIN.GENERATOR_MODEL != '':
            layout_generator.load_state_dict(torch.load(cfg.TRAIN.GENERATOR_MODEL))
        layout_optimizer = define_optimizers(layout_generator, cfg.GENERATOR.LR,
                                             cfg.GENERATOR.WEIGHT_DECAY)
        layout_generator.cuda()
        layout_evaluator.cuda()
        criterion = torch.nn.MSELoss()
        KL_crierion = torch.nn.KLDivLoss()
        criterion.cuda()
        KL_crierion.cuda()
        start_epoch = 0
        training_epoch, testing_epoch, training_error, testing_error = [], [], [], []
        for epoch in range(start_epoch, self.max_epoch):
            layout_generator.train()
            layout_evaluator.train()
            start_t = time.time()
            sample_num = cfg.TRAIN.SAMPLE_NUM
            label = torch.FloatTensor([[1.0]]).repeat(sample_num, 1).cuda()
            err_preds, restart, total_err, gt_room_std = 0, 1, 0, 0
            test_err_preds, test_restart, test_total_err = 0, 1, 0
            for step, data in enumerate(self.dataloader_train, 0):
                room_data, init_contour, gt_room_data, _ = data
                gt_layout_room = gt_room_data[0].transpose(0, 1).cuda()
                # room_data: [1, contour_type(step), sample_num, room_num, 3(point+direction)]
                # init_contour: [1, contour_type(step), sample_num, contour, 3]
                origin_layout_room = room_data[0].transpose(0, 1).cuda()
                room_num = origin_layout_room.shape[0]
                layout_init_contour = init_contour[0].transpose(0, 1).cuda()
                origin_layout = torch.cat((layout_init_contour, origin_layout_room),dim=0)
                refine_layout = layout_generator(origin_layout)
                refine_layout_room = refine_layout[-room_num:, :, :]
                refine_layout = torch.cat((layout_init_contour, refine_layout_room), dim=0).cuda()
                pred_score = layout_evaluator.forward(refine_layout)
                # get the refine room std and the origin room std
                room_std = torch.mean(torch.std(refine_layout_room, dim=0, unbiased=False), dim=1)
                room_std = room_std.reshape([len(room_std), 1]).cuda()
                pred_score = pred_score * (room_std > 0.05)
                if step == 299:
                    print("Test the score!\n")
                    gt_layout = torch.cat((layout_init_contour, gt_layout_room), dim=0).cuda()
                    gt_score = layout_evaluator.forward(gt_layout)
                    self.logger.info("In step 299, Pred_score1: {}, origin_score1: {}".format(pred_score[0], gt_score[0]))
                    self.logger.info("origin_layout0: {}".format(origin_layout_room[:, 0, :]))
                    self.logger.info("refine_layout0: {}".format(refine_layout_room[:, 0, :]))
                err_pred = criterion(pred_score, label) +  cfg.TRAIN.TAU * criterion(origin_layout_room, refine_layout_room)
                if step == 0 or restart:
                    err_preds = err_pred
                    restart = 0
                else:
                    err_preds = err_pred + err_preds
                if (step+1) % 25 == 0 and step != 0:
                    err_preds = err_preds / 25.0
                    if step+1 == 25:
                        total_err = err_preds.item()
                    else:
                        total_err = total_err + err_preds.item()
                    layout_optimizer.zero_grad()
                    err_preds.backward()
                    layout_optimizer.step()
                    restart = 1
            print('comparing total loss...')
            print('\033[1;31m current_epoch[{}] current_loss[{}] \033[0m \033[1;34m best_epoch[{}] best_loss[{}] \033[0m'.format(
                    epoch, total_err, self.best_epoch, self.best_loss))
            if epoch % 10 == 0:
                save_model(model=layout_generator, epoch=epoch, model_dir=self.model_dir, model_name='generator', best=False)
            # ================ #
            #      Testing     #
            # ================ #
            layout_generator.eval()
            layout_evaluator.eval()
            for step, data in enumerate(self.dataloader_test, 0):
                room_data, init_contour, _, _ = data
                # room_data: [1, contour_type(step), sample_num, room_num, 3(point+direction)]
                # init_contour: [1, contour_type(step), sample_num, contour, 3]
                origin_layout_room = room_data[0].transpose(0, 1).cuda()
                room_num = origin_layout_room.shape[0]
                layout_init_contour = init_contour[0].transpose(0, 1).cuda()
                origin_layout = torch.cat((layout_init_contour, origin_layout_room),dim=0)
                refine_layout = layout_generator(origin_layout)
                refine_layout_room = refine_layout[-room_num:, :, :]
                refine_layout = torch.cat((layout_init_contour, refine_layout_room), dim=0).cuda()
                pred_score = layout_evaluator.forward(refine_layout)
                # get the refine room std and the origin room std
                room_std = torch.mean(torch.std(refine_layout_room, dim=0, unbiased=False), dim=1)
                room_std = room_std.reshape([len(room_std), 1]).cuda()
                pred_score = pred_score * (room_std > 0.05)
                err_test_pred = criterion(pred_score, label) + cfg.TRAIN.TAU * criterion(origin_layout_room, refine_layout_room)
                if step == 0 or test_restart:
                    test_err_preds = err_test_pred.item()
                    test_restart = 0
                else:
                    test_err_preds = err_test_pred.item() + test_err_preds
                if (step+1) % 25 == 0 and step != 0:
                    test_err_preds = test_err_preds / 25.0
                    if step+1 == 25:
                        test_total_err = test_err_preds
                    else:
                        test_total_err = test_total_err + test_err_preds
                    test_restart = 1
            if test_total_err < self.best_loss:
                self.best_loss = test_total_err
                self.best_epoch = epoch
                print('saving best models...')
                save_model(model=layout_generator, epoch=epoch, model_dir=self.model_dir,
                           model_name='generator', best=True)
            print("test_total_err: ", test_total_err)
            # ================ #
            #      Saving      #
            # ================ #
            training_epoch.append(epoch)
            testing_epoch.append(epoch)
            training_error.append(total_err)
            testing_error.append(test_total_err*3)
            # plot
            plt.figure(0)
            plt.plot(training_epoch, training_error, color="r", linestyle="-", linewidth=1, label="training")
            plt.plot(testing_epoch, testing_error, color="b", linestyle="-", linewidth=1, label="testing")
            plt.xlabel("epoch")
            plt.ylabel("loss")
            plt.legend(loc='best')
            plt.savefig(os.path.join(self.output_dir, "loss.png"))
            plt.close(0)
            # loss
            with open(os.path.join(self.log_dir, 'train_log_loss.txt'), 'a') as f:
                f.write('{},{}\n'.format(epoch, training_error[-1]))
            with open(os.path.join(self.log_dir, 'test_log_loss.txt'), 'a') as f:
                f.write('{},{}\n'.format(epoch, testing_error[-1]))
            # print
            end_t = time.time()
            self.logger.info('[%d/%d] Loss_total: %.5f Test_loss: %.5f Time: %.2fs' % (epoch, self.max_epoch, total_err, test_total_err, end_t - start_t))

    def evaluate(self):
        eval_layout_evaluator, eval_layout_generator = self.define_models()
        if cfg.EVAL.MODEL_EVALUATOR == '':
            print("Please load the eval model path!")
            return 0
        else:
            eval_layout_evaluator.load_state_dict(torch.load(self.eval_model_path))
            eval_layout_generator.load_state_dict(torch.load(self.gen_model_path))
        eval_layout_evaluator.cuda()
        eval_layout_generator.cuda()
        eval_layout_evaluator.eval()
        eval_layout_generator.eval()
        layout_convertor = ConversionLayout()
        score_ratio_sum, score_area_sum, score_cost_sum, add_nums, loss_step, \
        refine_sum_hulls, init_sum_contour, test_num, random_seed = 0., 0., 0., 0, [], [], [], 1000, 100
        random.seed(random_seed)
        eval_metric = cfg.EVAL.EVAL_METRIC # 1 if evaluate metric else generate 3d building visualization
        for step, data in enumerate(self.dataloader_test, 0):
            room_data, init_contour, _, gt_room_data = data
            origin_layout_room = room_data[0].transpose(0, 1).cuda()
            room_num = origin_layout_room.shape[0]
            layout_init_contour = init_contour[0].transpose(0, 1).cuda()
            origin_layout = torch.cat((layout_init_contour, origin_layout_room), dim=0).cuda()
            refine_layout = eval_layout_generator(origin_layout)
            refine_layout_room = refine_layout[-room_num:, :, :]
            refine_layout = torch.cat((layout_init_contour, refine_layout_room), dim=0).cuda()
            pred_score1 = eval_layout_evaluator.forward(refine_layout)
            pred_score2 = eval_layout_evaluator.forward(origin_layout)
            # / ********** change the range of the score ************/
            pred_score1 = tran_score_range(pred_score1)
            pred_score2 = tran_score_range(pred_score2)
            # / ********** according to the pred_score to sort the refine layout room **********/
            sorted_index = torch.argsort(pred_score1.squeeze(), descending=True)
            # sorted_index = torch.argsort(pred_score2.squeeze(), descending=True)
            visual_refine_layout_rooms, refine_hulls = [], []
            for i in range(len(sorted_index)):
                visual_refine_layout_rooms.append(refine_layout_room[:, sorted_index[i], :])
            # /*************** data process **************/
            direction_dict = {0:[0, 0], 1:[0, 1], 2:[1, 0], 3:[1, 1]}
            visual_origin_layout_room = origin_layout_room[:, 0, :]
            for i in range(len(visual_origin_layout_room)):
                for j in range(len(direction_dict)):
                    if list(visual_origin_layout_room[i, 2:]) == direction_dict[j]:
                        visual_origin_layout_room[i, 2] = float(j)/4
                    for k in range(len(visual_refine_layout_rooms)):
                        if list(np.round(visual_refine_layout_rooms[k][i, 2:].detach().cpu().numpy())) == direction_dict[j]:
                            visual_refine_layout_rooms[k][i, 2] = float(j) / 4

            # /***************** save for 3d visualization and evluation metric*****************/
            max_num, num = 2, 0
            gt_hull = gt_room_data[0]
            print("step:", step)
            for k in range(len(visual_refine_layout_rooms)):
                refine_hull = layout_convertor.tran_layout_hull(layout_init_contour[:,0,:2].cpu(), visual_refine_layout_rooms[k][:, :3].cpu())
                if refine_hull != -1:
                    refine_hulls.append(refine_hull)
                    refine_sum_hulls.append(refine_hull)
                    init_sum_contour.append(np.array(init_contour[0][0][:, :2]*255, dtype=np.int))  
                    num = num + 1
                if not eval_metric:
                    if num == max_num:
                        break
            if not eval_metric:
                from miscc.render3D import render_3d_contour
                save_mesh_path = os.path.join(self.eval_dir, "layout_{}.ply".format(step))
                render_3d_contour(layout_init_contour[:, 0, :2].cpu(), gt_hull, refine_hulls, save_mesh_path)
        # /***************** for evaluation metrics *****************/
        if eval_metric:
            evaluation = evaluateMetric()
            index = [i for i in range(len(refine_sum_hulls))]
            random_index = random.sample(index, test_num)
            for num in range(len(refine_sum_hulls)):
                if num not in random_index:
                    continue
                rooms_hulls, init_hull, score_ratio, score_area, score_cost \
                    = refine_sum_hulls[num], init_sum_contour[num], 0., 0., 0.
                for i in range(len(rooms_hulls)):
                    score_room_ratio = evaluation.calculate_score_ratio(rooms_hulls[i])
                    score_room_area = evaluation.calculate_score_area(len(rooms_hulls), rooms_hulls[i], init_hull)
                    score_room_cost = evaluation.calculate_score_cost(rooms_hulls[i], init_hull)
                    score_ratio = score_ratio + score_room_ratio
                    score_area = score_area + score_room_area
                    score_cost = score_cost + score_room_cost
                score_ratio_sum = score_ratio_sum + score_ratio
                score_area_sum = score_area_sum + score_area
                score_cost_sum = score_cost_sum + score_cost
                add_nums = add_nums + 1
            print("{} layout can be evaluate! score_ratio: {}, score_area: {}, score_cost: {}".format(add_nums, score_ratio_sum/add_nums, score_area_sum/add_nums, score_cost_sum/add_nums))