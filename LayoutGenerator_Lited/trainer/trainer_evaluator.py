import os
import time
import json
import torch
import cv2 as cv
import numpy as np
import torch.optim as optim
from miscc.config import cfg
from miscc.utils import mkdir_p
import torch.backends.cudnn as cudnn
from model.model_LSTM import LayoutEvaluator
import matplotlib.pyplot as plt
plt.switch_backend('agg')


def visualization(point_hulls, pred_score, save_pic_path, contour_type=None, data_type="numpy", draw_text= False):
    """
    function: visualize the point hull to see the point
    """
    def visualize(boxes_collection):
        """
        function: visualize the line coord to see the visual image
        boxes_coord:[[],[],..] boxes_type:[]
        box_collection format as follow:
            box_collection: [(tensor([[boxes_coord],[boxes_coord],..]), [tensor,tensor,...])]
        example:[(tensor([[0.6235, 0.3686, 0.3373, 0.3686],
                  [0.3373, 0.3686, 0.3373, 0.6941],
                  [0.3373, 0.6941, 0.6235, 0.6941],
                  [0.6235, 0.6941, 0.6235, 0.3686],
                  [0.6235, 0.3686, 0.6235, 0.3686]]),
                  [tensor(0.), tensor(0.), tensor(0.), tensor(0.), tensor(0.)])]
        """
        from miscc import vutils
        background = np.ones((256, 256))
        im = vutils.save_bbox(background, boxes_collection, save_pic_path, normalize=True, draw_line=True, save=False)
        return im
    images = None
    hull_collection = []

    for i in range(len(point_hulls)):
        hull_coords = []
        hull_types = []
        if contour_type is None:
            contour_type = np.ones((len(point_hulls[i])))
        # data process on point hulls
        for n in range(len(point_hulls[i])):
            if n <= len(contour_type)-1 :
                type = contour_type[n] % 9
            else:
                type = 10.0
            hull_coord, hull_type = trans_hull_boxes_coord(point_hulls[i][n], type, data_type)
            for k in range(len(hull_coord)):
                hull_coords.append(hull_coord[k])
                hull_types.append(hull_type[k])
        # for visualization
        boxes_type = [torch.tensor(hull_types[t]) for t in range(len(hull_types))]
        boxes_coord = torch.tensor(hull_coords)
        hull_collection.append((boxes_coord, boxes_type))
    image = visualize(hull_collection)
    if draw_text:
        image = np.array(image.convert("RGB"))
        for i in range(len(pred_score)):
            text = str('%.4f' % pred_score[i])
            cv.putText(image, text, (15+256*i, 20), cv.FONT_HERSHEY_COMPLEX,
                       0.8, (0, 0, 0), thickness=2)
        cv.imwrite(save_pic_path, image)
    else:
        image.save(save_pic_path)


def trans_hull_boxes_coord(point_hull, boxes_type, data_type):
    # trans the point_hull[[]] to [[]]
    hull_coord = []
    hull_type = []
    if data_type == "tensor":
        point_hull = torch.squeeze(point_hull, 0)
    elif data_type == "numpy":
        point_hull = np.squeeze(point_hull, 0)
    for i in range(len(point_hull)):
        if i == len(point_hull) - 1:
            hull_coord.append([float('%.4f' % (float(point_hull[i][0]) / 256)), float('%.4f' % (float(point_hull[i][1]) / 256)),
                               float('%.4f' % (float(point_hull[0][0]) / 256)),
                               float('%.4f' % (float(point_hull[0][1]) / 256))])
        else:
            hull_coord.append([float('%.4f' % (float(point_hull[i][0]) / 256)), float('%.4f' % (float(point_hull[i][1]) / 256)),
                               float('%.4f' % (float(point_hull[i + 1][0]) / 256)),
                               float('%.4f' % (float(point_hull[i + 1][1]) / 256))])
        hull_type.append(boxes_type)
    return hull_coord, hull_type


def define_optimizers(model, lr, weight_decay):
    optimizer_model = optim.Adam(model.parameters(), lr=lr,
                                 weight_decay=weight_decay, betas=(0.5, 0.999))
    return optimizer_model


def save_model(model, epoch, model_dir, model_name, best=False):
    if best:
        torch.save(model.state_dict(), '%s/%s_best.pth' % (model_dir, model_name))
    else:
        torch.save(model.state_dict(), '%s/%s_%d.pth' % (model_dir, model_name, epoch))

def tran_score_range(score):
    max_range = max(score)
    min_range = min(score)
    score = (score - min_range)/(max_range - min_range) * 0.98
    return score

def calculate_contour_area(polygon):
    # calculate the area
    im = np.zeros((256, 256))
    polygon_mask = cv.fillPoly(im, [polygon], 255)
    area = np.sum(np.greater(polygon_mask, 0))
    return area

def room_size_range_hull(contour_hull, hull_type, range_type):
    # room size range 
    new_hull, hull_size = [], []
    for i in range(len(contour_hull)):
        size = calculate_contour_area(contour_hull[i])
        hull_size.append(size)
    hull_index = np.argsort(hull_size)
    hull_index = hull_index[::-1]
    for i in range(len(hull_index)):      
        new_hull.append(contour_hull[hull_index[i]])
    new_hull = np.array(new_hull)
    # # type range
    new_type, new_type_index = [], []
    for i in range(len(hull_type)):
        index = range_type.index(int(hull_type[i]%9))
        new_type_index.append(index)
    new_type_index = np.argsort(new_type_index)
    for i in range(len(new_type_index)):
        new_type.append(hull_type[new_type_index[i]])
    return new_hull, new_type

class LayoutTrainer(object):
    def __init__(self, output_dir, dataloader_train, dataloader_test, logger=None):
        self.output_dir = output_dir
        self.model_dir = os.path.join(output_dir, 'Model')
        self.log_dir = os.path.join(output_dir, 'Log')
        self.eval_dir = os.path.join(output_dir, 'eval')
        self.model_path = os.path.join(output_dir, cfg.EVAL.MODEL_EVALUATOR)
        mkdir_p(self.model_dir)
        mkdir_p(self.log_dir)
        mkdir_p(self.eval_dir)
        # save the information of cfg
        log_cfg = os.path.join(self.log_dir, 'log_cfg.json')
        self.logger = logger
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
                torch.nn.init.xavier_normal_(m.weight)
                torch.nn.init.constant_(m.bias, 0)
        room_dim = 4
        room_hiddern_dim = cfg.TRAIN.ROOM_HIDDERN_DIM
        score_hiddern_dim = cfg.TRAIN.SCORE_HIDDERN_DIM
        evaluator = LayoutEvaluator(room_dim, room_hiddern_dim, score_hiddern_dim, bidirectional=False)
        evaluator.apply(weight_init)
        return evaluator

    def train(self):
        training_epoch = []
        testing_epoch = []
        training_error = []
        testing_error = []
        layout_evaluator = self.define_models()
        if cfg.TRAIN.EVALUATOR != '':
            layout_evaluator.load_state_dict(torch.load(cfg.TRAIN.EVALUATOR))
        layout_optimizer = define_optimizers(layout_evaluator, cfg.EVALUATOR.LR,
                                             cfg.EVALUATOR.WEIGHT_DECAY)
        self.logger.info("layout_evaluator: {}".format(layout_evaluator))
        criterion = torch.nn.HingeEmbeddingLoss(margin=cfg.TRAIN.MARGIN)
        criterion.cuda()
        layout_evaluator.cuda()
        start_epoch = 0
        for epoch in range(start_epoch, self.max_epoch):
            layout_evaluator.train()
            start_t = time.time()
            sample_num = cfg.TRAIN.SAMPLE_NUM
            scale = sample_num * (sample_num - 1) // 2
            train_label = torch.FloatTensor([[-1.0]]).cuda()
            restart, err_preds, total_err  = 1, 0.0, 0.0
            for step, data in enumerate(self.dataloader_train, 0):
                pair_data, init_contour, _, _, indicator_values = data
                # data: [1, 2500, 2, 4, 3] init_contour: [2500, 2, n, 3]
                # indicator_hull: [1, 2500, 2, 5, n, 2] indicator_value[:800, :]
                layout1_room, layout2_room = pair_data[0][0], pair_data[0][1]
                layout1_init_contour, layout2_init_contour = init_contour[0], init_contour[0]
                layout1 = torch.cat((layout1_init_contour, layout1_room), dim=1).transpose(0, 1).cuda()
                layout2 = torch.cat((layout2_init_contour, layout2_room), dim=1).transpose(0, 1).cuda()
                pred_score1 = layout_evaluator(layout1)
                pred_score2 = layout_evaluator(layout2)
                # indicators_values = torch.from_numpy(np.array(indicator_values[0])).type(torch.float32).cuda()
                indicator_values = indicator_values[0].reshape([len(indicator_values[0]), 1]).float().cuda()
                scale = indicator_values.shape[0]
                pred = torch.sum((pred_score1 - pred_score2) * indicator_values) / scale
                if step == 299:
                    self.logger.info("In epoch {}, pred_score1: {}, pred_score2: {}".format(epoch, pred_score1[0], pred_score2[0]))
                    self.logger.info("pred: {}".format(pred))
                err_pred = criterion(pred, train_label)
                if step == 0 or restart:
                    err_preds = err_pred
                    restart = 0
                else:
                    err_preds = err_pred + err_preds
                if (step+1) % 50 == 0 and step != 0:
                    err_preds = err_preds / 50.0
                    total_err = total_err + err_preds.item()
                    layout_optimizer.zero_grad()
                    err_preds.backward()
                    layout_optimizer.step()
                    restart = 1
            print('comparing total loss...')
            print('\033[1;31m current_epoch[{}] current_loss[{}] \033[0m \033[1;34m best_epoch[{}] best_loss[{}] \033[0m'.format(epoch, total_err, self.best_epoch, self.best_loss))
            if epoch % 10 == 0:
                save_model(model=layout_evaluator, epoch=epoch, model_dir=self.model_dir,
                           model_name='evaluator', best=False)
            # ================ #
            #      testing     #
            # ================ #
            with torch.no_grad():
                layout_evaluator.eval()
                test_label = torch.FloatTensor([[-1.0]]).cuda()
                test_restart, test_err_preds, test_total_err = 1, 0.0, 0.0
                for step, data in enumerate(self.dataloader_test, 0):
                    test_pair_data, test_init_contour, _, _, test_indicator_values = data
                    test_layout1_room, test_layout2_room = test_pair_data[0][0], test_pair_data[0][1]
                    test_layout1_init_contour, test_layout2_init_contour = test_init_contour[0], test_init_contour[0]
                    test_layout1 = torch.cat((test_layout1_init_contour, test_layout1_room), dim=1).transpose(0, 1).cuda()
                    test_layout2 = torch.cat((test_layout2_init_contour, test_layout2_room), dim=1).transpose(0, 1).cuda()
                    test_pred_score1 = layout_evaluator.forward(test_layout1)
                    test_pred_score2 = layout_evaluator.forward(test_layout2)
                    test_indicator_values = test_indicator_values[0].reshape([len(test_indicator_values[0]), 1]).float().cuda()
                    scale = test_indicator_values.shape[0]
                    test_pred = torch.sum((test_pred_score1 - test_pred_score2)* test_indicator_values)/ scale
                    test_err_pred = criterion(test_pred, test_label)
                    if step == 0 or test_restart:
                        test_err_preds = test_err_pred
                        test_restart = 0
                    else:
                        test_err_preds = test_err_preds + test_err_pred
                    if (step+1) % 50 == 0 and step != 0:
                        test_err_preds = test_err_preds / 50.0
                        test_total_err = test_total_err + test_err_preds
                        test_restart = 1
                    if test_total_err < self.best_loss:
                        self.best_loss = test_total_err
                        self.best_epoch = epoch
                        print('saving best models...')
                        save_model(model=layout_evaluator, epoch=epoch, model_dir=self.model_dir,
                                model_name='evaluator', best=True)
            # ================ #
            #      Saving      #
            # ================ #
            training_epoch.append(epoch)
            training_error.append(total_err/3.0)
            testing_epoch.append(epoch)
            testing_error.append(test_total_err)
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
            with open(os.path.join(self.log_dir, 'Train_log_loss.txt'), 'a') as f:
                f.write('{},{}\n'.format(epoch, training_error[-1]))
            with open(os.path.join(self.log_dir, 'Test_log_loss.txt'), 'a') as f:
                f.write('{},{}\n'.format(epoch, testing_error[-1]))
            # print
            end_t = time.time()
            self.logger.info(
                '[{}/{}] Train_Loss_total: {:.5} Test_Loss_total: {:.5} Time: {:.5}s'.format(epoch, self.max_epoch, total_err, test_total_err, end_t - start_t))

    def evaluate(self):
        eval_layout_evaluator = self.define_models()
        if cfg.EVAL.MODEL_EVALUATOR == '':
            print("Please load the eval model path!")
        else:
            eval_layout_evaluator.load_state_dict(torch.load(self.model_path))
        eval_layout_evaluator.cuda()
        eval_layout_evaluator.eval()
        num = 0
        test_index = cfg.EVAL.TEST_INDEX
        save_score_path = os.path.join(self.eval_dir, "layout.txt")
        room_classes = ['livingroom', 'bedroom', 'corridor', 'kitchen',
                        'washroom', 'study', 'closet', 'storage', 'balcony']
        room_classes_size_range = [6, 4, 8, 7, 2, 5, 3, 1, 0] # according to the class
        room_classes_size_range = room_classes_size_range[::-1]
        with open(save_score_path, "w") as score_file:
            for step, data in enumerate(self.dataloader_test, 0):
                num = num + 1
                print("step: ", step)
                eval_pair_data, eval_init_contour, eval_pair_hulls, contour_types, _ = data
                
                eval_layout1_room, eval_layout2_room = eval_pair_data[0][0], eval_pair_data[0][1]
                eval_layout1_init_contour, eval_layout2_init_contour = eval_init_contour[0], \
                                                                       eval_init_contour[0]
                eval_layout1 = torch.cat((eval_layout1_init_contour, eval_layout1_room), dim=1).transpose(0, 1).cuda()
                eval_layout2 = torch.cat((eval_layout2_init_contour, eval_layout2_room), dim=1).transpose(0, 1).cuda()
                eval_score1 = eval_layout_evaluator.forward(eval_layout1)
                eval_score2 = eval_layout_evaluator.forward(eval_layout2)
                eval_score1 = tran_score_range(eval_score1)
                eval_score2 = tran_score_range(eval_score2)
                save_num = 10
                for i in range(save_num):
                    eval_layout1_hull, eval_layout2_hull = eval_pair_hulls[0][i][0], eval_pair_hulls[0][i][1]
                    eval_layout1_hull, new_contour_type = room_size_range_hull(eval_layout1_hull, contour_types[0], room_classes_size_range)
                    eval_layout2_hull, new_contour_type = room_size_range_hull(eval_layout2_hull, contour_types[0], room_classes_size_range)
                    pred_score = [eval_score1[i][0], eval_score2[i][0]]
                    save_path = os.path.join(self.eval_dir, "layout_score_{}_{}.png".format(num, i))
                    visualization([eval_layout1_hull, eval_layout2_hull], pred_score, save_path, new_contour_type, "none", draw_text=True)
                # /************** for save the data description *****************/
                # print("In layout_num: {}".format(num), "layout1_score: {:.2}".format(eval_score1[0][0]),
                #       "layout2_score: {:.2}".format(eval_score2[0][0]), file=score_file)
                # room_classes = ['livingroom', 'bedroom', 'corridor', 'kitchen',
                #                 'washroom', 'study', 'closet', 'storage', 'balcony']
                # room_count = []
                # for i in range(len(room_classes)):
                #     room_num = contour_types[0].count(i) + contour_types[0].count(i+9*1) + \
                #                contour_types[0].count(i+9*2) + contour_types[0].count(i+9*3)
                #     room_count.append(room_num)
                # print("contour_types:", contour_types)
                # if np.sum(room_count) != len(contour_types[0]):
                #     print("error happened!")
                #     return 0
                # score_file.writelines("In layout_num: {} It has ".format(num))
                # for i in range(len(room_classes)):
                #     if room_count[i] != 0:
                #         score_file.writelines(" {} {}".format(room_count[i], room_classes[i]))
                # score_file.writelines("\n")

