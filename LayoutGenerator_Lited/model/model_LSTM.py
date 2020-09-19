import torch
import torch.nn as nn
from miscc.config import cfg


class LayoutEvaluator(nn.Module):
    def __init__(self, room_dim, room_hiddern_dim, score_hiddern_dim, bidirectional=True):
        super(LayoutEvaluator, self).__init__()
        self.LayoutSeq = nn.LSTM(input_size=room_dim, hidden_size=room_hiddern_dim, bidirectional=bidirectional) #, batch_first=True)
        if bidirectional:
            score_dim = [2*room_hiddern_dim, score_hiddern_dim, 1]
        else:
            score_dim = [room_hiddern_dim, score_hiddern_dim, 1]
        self.mlp = self.build_mlp(dim_list=score_dim)
        # if cfg.TRAIN.GENERATOR:
        #     for p in self.parameters():
        #         p.require_grad = False

    def build_mlp(self, dim_list):
        layers = []
        for i in range(len(dim_list)-1):
            dim_in, dim_out = dim_list[i], dim_list[i+1]
            layers.append(nn.Linear(dim_in, dim_out))
            if i + 1 == len(dim_list) - 1:
                layers.append(nn.BatchNorm1d(dim_out))
                layers.append(nn.Sigmoid())
                # continue
            else:
                layers.append(nn.BatchNorm1d(dim_out))
                layers.append(nn.ReLU())
                # continue
        return nn.Sequential(*layers)

    def forward(self, input):
        # input: (room_num, batchsize, input_dim)
        layout_feature, _ = self.LayoutSeq(input)
        layout_feature = layout_feature[-1]
        Score = self.mlp(layout_feature)
        return Score


class LayoutGenerator(nn.Module):
    def __init__(self, room_dim, room_hiddern_dim, room_gen_hiddern_dim, init_hidden_dim, max_len, logger=None, bidirectional=False):
        super(LayoutGenerator, self).__init__()
        # self.hidden_process = self.build_process(dim_list=init_hidden_dim)
        self.encoder = nn.LSTM(input_size=room_dim, hidden_size=room_hiddern_dim, bidirectional=bidirectional)
        self.decoder = nn.LSTM(input_size=room_hiddern_dim, hidden_size=room_hiddern_dim, bidirectional=False)
        for name, param in self.encoder.named_parameters():
            if 'bias' in name:
                nn.init.constant(param, 0.01)
            elif 'weight' in name:
                # nn.init.constant(param, 0.1)
                nn.init.orthogonal(param)
        # for name, param in self.decoder.named_parameters():
        #     if 'bias' in name:
        #         nn.init.constant(param, 0.01)
        #     elif 'weight' in name:
        #         # nn.init.constant(param, 0.1)
        #         nn.init.xavier_normal(param)
        gen_dim = [room_hiddern_dim, room_gen_hiddern_dim, 4]
        # room_hiddern_dim * 2 if bidirectional else room_hiddern_dim
        self.mlp = self.build_mlp(dim_list=gen_dim)
        self.logger = logger

    def build_process(self, dim_list):
        layers = []
        for i in range(len(dim_list) - 1):
            dim_in, dim_out = dim_list[i], dim_list[i + 1]
            layers.append(nn.Linear(dim_in, dim_out))
            layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    def build_mlp(self, dim_list):
        layers = []
        for i in range(len(dim_list)-1):
            dim_in, dim_out = dim_list[i], dim_list[i+1]
            layers.append(nn.Linear(dim_in, dim_out))
            if i + 1 == len(dim_list) - 1:
                layers.append(nn.Sigmoid())
            else:
                layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    def _cat_directions(self, h):
        """ If the encoder is bidirectional, do the following transformation.
            (#directions * #layers, #batch, hidden_size) -> (#layers, #batch, #directions * hidden_size)
        """
        h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
        return h

    def forward(self, room):
        # hiddern: turn room[4, 2, 3] --> [1, 2, 4]
        # hidden = torch.unsqueeze(torch.squeeze(hidden), dim=0)

        encoder_outputs, encoder_hidden = self.encoder(room)

        # self.logger.info("encoder_outputs: {}, \n encoder_hidden: {}".format(encoder_outputs, encoder_hidden))
        # # encoder_hidden = tuple([self._cat_directions(h) for h in encoder_hidden])

        # layout_feature, _ = self.decoder(encoder_outputs)

        # self.logger.info("layout_feature: {}".format(layout_feature))

        # encoder_outputs = torch.reshape(torch.squeeze(encoder_outputs), (cfg.TRAIN.SAMPLE_NUM, 4, 3))
        layout_feature = encoder_outputs.transpose(0, 1)
        # layout_feature = layout_feature.transpose(0, 1)
        room_point = self.mlp(layout_feature)
        room_point = room_point.transpose(0, 1)
        return room_point


if __name__ == "__main__":
    data = torch.randn(4, 2, 3)
    # print(data)
    room_dim = 3
    room_hiddern_dim = 16
    score_hiddern_dim = 32
    gen_room_hiddern_dim = 16
    max_len = 50
    # layout_evaluator = LayoutEvaluator(room_dim, room_hiddern_dim, score_hiddern_dim, bidirectional=False)
    # layout_evaluator = layout_evaluator.cuda()
    # output = layout_evaluator(data)
    # print(output)
    layout_generator = LayoutGenerator(room_dim, room_hiddern_dim,
                                       gen_room_hiddern_dim, [3, 1], max_len, bidirectional=False)
    output = layout_generator.forward(data)
    print(output.shape)
