import subprocess
import os 
import torch
import torch.nn as nn
from torch.autograd import Variable

from models.yolo_layer_2 import YOLOLayer
from models.resblock import resblock
from utils.parse_yolo_weights import parse_yolo_weights
from utils.lossdict import Loss_Dict

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
        if self.cfg['MODEL']['LOAD_WEIGHTS']:
            self.parse_weights()
        self.loss_dict = Loss_Dict()
        self.loss_dict.init(['total','xy', 'wh', 'conf', 'cls', 'l2'])

    def add_conv(self,in_ch, out_ch, ksize, stride):
        """
        Add a conv2d / batchnorm / leaky ReLU block.
        Args:
            in_ch (int): number of input channels of the convolution layer.
            out_ch (int): number of output channels of the convolution layer.
            ksize (int): kernel size of the convolution layer.
            stride (int): stride of the convolution layer.
        Returns:
            stage (Sequential) : Sequential layers composing a convolution block.
        """
        stage = nn.Sequential()
        pad = (ksize - 1) // 2
        stage.add_module('conv', nn.Conv2d(in_channels=in_ch,
                                           out_channels=out_ch, kernel_size=ksize, stride=stride,
                                           padding=pad, bias=False))
        stage.add_module('batch_norm', nn.BatchNorm2d(out_ch))
        stage.add_module('leaky', nn.LeakyReLU(0.1))
        return stage

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
        mlist.append(self.add_conv(in_ch=3, out_ch=32, ksize=3, stride=1)) #0
        mlist.append(self.add_conv(in_ch=32, out_ch=64, ksize=3, stride=2)) #1
        mlist.append(resblock(ch=64)) #2
        mlist.append(self.add_conv(in_ch=64, out_ch=128, ksize=3, stride=2)) #3
        mlist.append(resblock(ch=128, nblocks=2)) #4
        mlist.append(self.add_conv(in_ch=128, out_ch=256, ksize=3, stride=2)) #5
        mlist.append(resblock(ch=256, nblocks=8))    # shortcut 1 from here #6
        mlist.append(self.add_conv(in_ch=256, out_ch=512, ksize=3, stride=2)) #7
        mlist.append(resblock(ch=512, nblocks=8))    # shortcut 2 from here #8
        mlist.append(self.add_conv(in_ch=512, out_ch=1024, ksize=3, stride=2)) #9
        mlist.append(resblock(ch=1024, nblocks=4)) #10

        # YOLOv3
        mlist.append(resblock(ch=1024, nblocks=2, shortcut=False)) #11
        mlist.append(self.add_conv(in_ch=1024, out_ch=512, ksize=1, stride=1)) #12
        # 1st yolo branch
        mlist.append(self.add_conv(in_ch=512, out_ch=1024, ksize=3, stride=1)) #13
        mlist.append(YOLOLayer(cfg, layer_no=0, in_ch=1024)) #14

        mlist.append(self.add_conv(in_ch=512, out_ch=256, ksize=1, stride=1)) #15
        mlist.append(nn.Upsample(scale_factor=2, mode='nearest')) #16
        mlist.append(self.add_conv(in_ch=768, out_ch=256, ksize=1, stride=1)) #17
        mlist.append(self.add_conv(in_ch=256, out_ch=512, ksize=3, stride=1)) #18
        mlist.append(resblock(ch=512, nblocks=1, shortcut=False)) #19
        mlist.append(self.add_conv(in_ch=512, out_ch=256, ksize=1, stride=1)) #20
        # 2nd yolo branch
        mlist.append(self.add_conv(in_ch=256, out_ch=512, ksize=3, stride=1)) #21
        mlist.append(YOLOLayer(cfg, layer_no=1, in_ch=512)) #22

        mlist.append(self.add_conv(in_ch=256, out_ch=128, ksize=1, stride=1)) #23
        mlist.append(nn.Upsample(scale_factor=2, mode='nearest')) #24
        mlist.append(self.add_conv(in_ch=384, out_ch=128, ksize=1, stride=1)) #25
        mlist.append(self.add_conv(in_ch=128, out_ch=256, ksize=3, stride=1)) #26
        mlist.append(resblock(ch=256, nblocks=2, shortcut=False)) #27
        mlist.append(YOLOLayer(cfg,layer_no=2, in_ch=256)) #28
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
            x = Variable(x.type(dtype))
            targets = Variable(targets.type(dtype), requires_grad=False)
        else:
            output = []
        route_layers = []
        for i, module in enumerate(self.module_list):
            # yolo layers
            if i in [14, 22, 28]:
                if train:
                    x,loss_xy,loss_wh,loss_obj,loss_cls,loss_l2= module(x, targets)
                    self.loss_dict.update(losses=[x,loss_xy,loss_wh,loss_obj,loss_cls,loss_l2])
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
        if weights_path is None:
            subprocess.call('./requirements/download_weights.sh')
            weights_path = './weights/yolov3.weights'
            print('Setting path to weights file as-',weights_path)
        assert os.path.exists(weights_path)
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


