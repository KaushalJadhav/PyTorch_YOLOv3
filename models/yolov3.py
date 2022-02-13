import torch
import torch.nn as nn
from torch.autograd import Variable
from collections import defaultdict

from models.yolo_layer import YOLOLayer
from models.resblock import *
from utils.parse_yolo_weights import parse_yolo_weights

class YOLOv3(nn.Module):
    """
    YOLOv3 model module. The module list is defined by create_yolov3_modules function. \
    The network returns loss values from three YOLO layers during training \
    and detection results during test.
    """
    def __init__(self,cfg,):
        """
        Initialization of YOLOv3 class.
        Args:
            cfg (dict): Configuration file used in YOLOLayer.
        """
        super(YOLOv3, self).__init__()
        self.cfg = cfg
        if self.cfg['MODEL']['TYPE'].upper() == 'YOLOV3'.upper():
            self.module_list = self.create_yolov3_modules(cfg)
        else:
            raise Exception('Model name {} is not available'.format(cfg['MODEL']['TYPE']))
        self.parse_weights()

    def create_yolov3_modules(self,cfg):
        """
        Build yolov3 layer modules.
        Args:
            cfg (dict): Configuration file.
            See YOLOLayer class for details.
        Returns:
            mlist (ModuleList): YOLOv3 module list.
        """

        # DarkNet53
        mlist = nn.ModuleList()
        mlist.append(add_conv(in_ch=3, out_ch=32, ksize=3, stride=1))
        mlist.append(add_conv(in_ch=32, out_ch=64, ksize=3, stride=2))
        mlist.append(resblock(ch=64))
        mlist.append(add_conv(in_ch=64, out_ch=128, ksize=3, stride=2))
        mlist.append(resblock(ch=128, nblocks=2))
        mlist.append(add_conv(in_ch=128, out_ch=256, ksize=3, stride=2))
        mlist.append(resblock(ch=256, nblocks=8))    # shortcut 1 from here
        mlist.append(add_conv(in_ch=256, out_ch=512, ksize=3, stride=2))
        mlist.append(resblock(ch=512, nblocks=8))    # shortcut 2 from here
        mlist.append(add_conv(in_ch=512, out_ch=1024, ksize=3, stride=2))
        mlist.append(resblock(ch=1024, nblocks=4))

        # YOLOv3
        mlist.append(resblock(ch=1024, nblocks=2, shortcut=False))
        mlist.append(add_conv(in_ch=1024, out_ch=512, ksize=1, stride=1))
        # 1st yolo branch
        mlist.append(add_conv(in_ch=512, out_ch=1024, ksize=3, stride=1))
        mlist.append(YOLOLayer(cfg, layer_no=0, in_ch=1024))

        mlist.append(add_conv(in_ch=512, out_ch=256, ksize=1, stride=1))
        mlist.append(nn.Upsample(scale_factor=2, mode='nearest'))
        mlist.append(add_conv(in_ch=768, out_ch=256, ksize=1, stride=1))
        mlist.append(add_conv(in_ch=256, out_ch=512, ksize=3, stride=1))
        mlist.append(resblock(ch=512, nblocks=1, shortcut=False))
        mlist.append(add_conv(in_ch=512, out_ch=256, ksize=1, stride=1))
        # 2nd yolo branch
        mlist.append(add_conv(in_ch=256, out_ch=512, ksize=3, stride=1))
        mlist.append(YOLOLayer(cfg, layer_no=1, in_ch=512))

        mlist.append(add_conv(in_ch=256, out_ch=128, ksize=1, stride=1))
        mlist.append(nn.Upsample(scale_factor=2, mode='nearest'))
        mlist.append(add_conv(in_ch=384, out_ch=128, ksize=1, stride=1))
        mlist.append(add_conv(in_ch=128, out_ch=256, ksize=3, stride=1))
        mlist.append(resblock(ch=256, nblocks=2, shortcut=False))
        mlist.append(YOLOLayer(cfg,layer_no=2, in_ch=256))
        return mlist

    def forward(self,x,targets=None,cuda=True):
        """
        Forward path of YOLOv3.
        Args:
            x (torch.Tensor) : input data whose shape is :math:`(N, C, H, W)`, \
                where N, C are batchsize and num. of channels.
            targets (torch.Tensor) : label array whose shape is :math:`(N, 50, 5)`

        Returns:
            training:
                output (torch.Tensor): loss tensor for backpropagation.
            test:
                output (torch.Tensor): concatenated detection results.
        """
        train = targets is not None
        if train:
            dtype = torch.cuda.FloatTensor if cuda else torch.FloatTensor
            imgs = Variable(imgs.type(dtype))
            targets = Variable(targets.type(dtype), requires_grad=False)
        output = []
        self.loss_dict = defaultdict(float)
        route_layers = []
        for i, module in enumerate(self.module_list):
            # yolo layers
            if i in [14, 22, 28]:
                if train:
                    x, *loss_dict = module(x, targets)
                    for name, loss in zip(['xy', 'wh', 'conf', 'cls', 'l2'] , loss_dict):
                        self.loss_dict[name] += loss
                else:
                    x = module(x)
                output.append(x)
            else:
                x = module(x)

            # route layers
            if i in [6, 8, 12, 20]:
                route_layers.append(x)
            if i == 14:
                x = route_layers[2]
            if i == 22:  # yolo 2nd
                x = route_layers[3]
            if i == 16:
                x = torch.cat((x, route_layers[1]), 1)
            if i == 24:
                x = torch.cat((x, route_layers[0]), 1)
        if train:
            return sum(output)
        else:
            return torch.cat(output, 1)
    
    def parse_weights(self):
        weights_path = self.cfg['MODEL']['WEIGHTS_PATH']
        if weights_path is not None:
            print("loading darknet weights from ....", weights_path)
            parse_yolo_weights(self,weights_path)
    
    def load_state(self,state):
        if 'model_state_dict' in state.keys():
            self.load_state_dict(state['model_state_dict'])
        else:
            self.load_state_dict(state)
    
    def get_params(self):
        # set weight decay only on conv.weight
        batch_size = self.cfg['DATA']['BATCHSIZE']
        subdivision = self.cfg['DATA']['SUBDIVISION']
        decay = self.cfg['SOLVER']['DECAY']
        params_dict = dict(self.named_parameters())
        params = []
        for key, value in params_dict.items():
            if 'conv.weight' in key:
                params += [{'params':value, 'weight_decay':decay * batch_size * subdivision}]
            else:
                params += [{'params':value, 'weight_decay':0.0}]
        return params 


