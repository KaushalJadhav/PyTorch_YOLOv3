import torch
import torch.nn as nn
import numpy as np
from utils.utils import bboxes_iou


class YOLOLayer(nn.Module):
    """
    detection layer corresponding to yolo_layer.c of darknet
    """
    def __init__(self, cfg,layer_no, in_ch,):
        """
        Args:
            cfg (dict) : Configuration file.
                ANCHORS (list of tuples) :
                ANCH_MASK:  (list of int list): index indicating the anchors to be
                    used in YOLO layers. One of the mask group is picked from the list.
                N_CLASSES (int): number of classes
            layer_no (int): YOLO layer number - one from (0, 1, 2).
            in_ch (int): number of input channels.
            ignore_thre (float): threshold of IoU above which objectness training is ignored.
        """

        super(YOLOLayer, self).__init__()
        self.strides = [32, 16, 8] # fixed
        self.stride = self.strides[layer_no]

        self.cfg = cfg 
        self.n_classes = self.cfg['MODEL']['N_CLASSES']
        self.n_ch = 5 + self.n_classes
        self.ignore_thre = self.cfg['TRAIN']['IGNORETHRE']
        self.batchsize = self.cfg['DATA']['BATCHSIZE']
        
        self.dtype=torch.FloatTensor
        self.anchors = self.cfg['MODEL']['ANCHORS']
        self.anch_mask = self.cfg['MODEL']['ANCH_MASK'][layer_no]
        self.n_anchors = len(self.anch_mask)
        self.all_anchors_grid = [(w / self.stride, h / self.stride)
                                 for w, h in self.anchors]
        self.masked_anchors = [self.all_anchors_grid[i]
                               for i in self.anch_mask]
        
        self.conv = nn.Conv2d(in_channels=in_ch,
                              out_channels=self.n_anchors * (self.n_ch),
                              kernel_size=1, stride=1, padding=0)
        
        self.Lambda = 0.5
    
    def bce_loss(self,weight=None):
        return nn.BCELoss(weight=weight,size_average=False)
    
    @property
    def l2_loss(self):
        return nn.MSELoss(size_average=False)
    
    def get_pred(self,fsize,output):
        x_shift = self.dtype(np.broadcast_to(np.arange(fsize, dtype=np.float32), output.shape[:4]))
        y_shift = self.dtype(np.broadcast_to(np.arange(fsize, dtype=np.float32).reshape(fsize, 1), output.shape[:4]))
        masked_anchors = np.array(self.masked_anchors)
        w_anchors = self.dtype(np.broadcast_to(np.reshape(masked_anchors[:, 0], (1, self.n_anchors, 1, 1)), output.shape[:4]))
        h_anchors = self.dtype(np.broadcast_to(np.reshape(masked_anchors[:, 1], (1, self.n_anchors, 1, 1)), output.shape[:4]))

        pred = output.clone()
        pred[..., 0] += x_shift
        pred[..., 1] += y_shift
        pred[..., 2] = torch.exp(pred[..., 2]) * w_anchors
        pred[..., 3] = torch.exp(pred[..., 3]) * h_anchors
        return pred
    
    def get_output(self,xin):
        output = self.conv(xin)
        fsize = output.shape[2]
        output = output.view(self.batchsize, self.n_anchors, self.n_ch, fsize, fsize)
        output = output.permute(0, 1, 3, 4, 2)  # .contiguous()
        # logistic activation for xy, obj, cls
        output[..., np.r_[:2, 4:self.n_ch]] = torch.sigmoid(output[..., np.r_[:2, 4:self.n_ch]])
        return output,fsize
    
    def get_best_mask(self,truth_box):
        ref_anchors = np.zeros((len(self.all_anchors_grid), 4))
        ref_anchors[:, 2:] = np.array(self.all_anchors_grid)
        ref_anchors = torch.FloatTensor(ref_anchors)
        anchor_ious_all = bboxes_iou(truth_box.cpu(),ref_anchors)
        best_n_all = np.argmax(anchor_ious_all, axis=1)
        best_n_mask = ((best_n_all == self.anch_mask[0]) | (best_n_all == self.anch_mask[1]) | (best_n_all == self.anch_mask[2]))
        return best_n_all,best_n_mask
    
    def get_best_iou(self,pred,truth_box):
        pred_ious = bboxes_iou(pred.contiguous().view(-1, 4), truth_box, xyxy=False)
        pred_best_iou, _ = pred_ious.max(dim=1)
        pred_best_iou = (pred_best_iou > self.ignore_thre)
        pred_best_iou = pred_best_iou.view(pred.shape[:3])
        return pred_best_iou
    
    def process(self,x,obj_mask,tgt_mask,tgt_scale):
        x[..., 4] *= obj_mask
        x[..., np.r_[0:4, 5:self.n_ch]] *= tgt_mask
        x[..., 2:4] *= tgt_scale
        return x
    
    def get_losses(self,output,target,obj_mask,tgt_mask,tgt_scale):
        output= self.process(output,obj_mask,tgt_mask,tgt_scale)
        target= self.process(target,obj_mask,tgt_mask,tgt_scale)
        wbce = self.bce_loss(weight=tgt_scale*tgt_scale)  # weighted BCEloss
        bce = self.bce_loss()
        loss_xy = wbce(output[..., :2], target[..., :2])
        loss_wh = self.l2_loss(output[..., 2:4], target[..., 2:4])*self.Lambda
        loss_obj = bce(output[..., 4], target[..., 4])
        loss_cls = bce(output[..., 5:], target[..., 5:])
        loss_l2 = self.l2_loss(output, target)
        loss = loss_xy + loss_wh + loss_obj + loss_cls
        return loss, loss_xy, loss_wh, loss_obj, loss_cls, loss_l2

    def forward(self, xin, labels=None):
        """
        In this
        Args:
            xin (torch.Tensor): input feature map whose size is :math:`(N, C, H, W)`, \
                where N, C, H, W denote batchsize, channel width, height, width respectively.
            labels (torch.Tensor): label data whose size is :math:`(N, K, 5)`. \
                N and K denote batchsize and number of labels.
                Each label consists of [class, xc, yc, w, h]:
                    class (float): class index.
                    xc, yc (float) : center of bbox whose values range from 0 to 1.
                    w, h (float) : size of bbox whose values range from 0 to 1.
        Returns:
            loss (torch.Tensor): total loss - the target of backprop.
            loss_xy (torch.Tensor): x, y loss - calculated by binary cross entropy (BCE) \
                with boxsize-dependent weights.
            loss_wh (torch.Tensor): w, h loss - calculated by l2 without size averaging and \
                with boxsize-dependent weights.
            loss_obj (torch.Tensor): objectness loss - calculated by BCE.
            loss_cls (torch.Tensor): classification loss - calculated by BCE for each class.
            loss_l2 (torch.Tensor): total l2 loss - only for logging.
        """
        self.dtype = torch.cuda.FloatTensor if xin.is_cuda else torch.FloatTensor
        output,fsize = self.get_output(xin)
    
        # calculate pred - xywh obj cls
        pred = self.get_pred(fsize,output)

        if labels is None:  # not training
            pred[..., :4] *= self.stride
            return pred.contiguous().view(self.batchsize, -1, self.n_ch).data

        pred = pred[..., :4].data

        # target assignment

        tgt_mask = torch.zeros(self.batchsize, self.n_anchors,
                               fsize, fsize, 4 + self.n_classes).type(self.dtype)
        obj_mask = torch.ones(self.batchsize, self.n_anchors,
                              fsize, fsize).type(self.dtype)
        tgt_scale = torch.zeros(self.batchsize, self.n_anchors,
                                fsize, fsize, 2).type(self.dtype)

        target = torch.zeros(self.batchsize, self.n_anchors,
                             fsize, fsize, self.n_ch).type(self.dtype)

        labels = labels.cpu().data
        nlabel = (labels.sum(dim=2) > 0).sum(dim=1)  # number of objects

        truth_x_all = labels[:, :, 1] * fsize
        truth_y_all = labels[:, :, 2] * fsize
        truth_w_all = labels[:, :, 3] * fsize
        truth_h_all = labels[:, :, 4] * fsize
        truth_i_all = truth_x_all.to(torch.int16).numpy()
        truth_j_all = truth_y_all.to(torch.int16).numpy()

        for b in range(self.batchsize):
            n = int(nlabel[b])
            if n == 0:
                continue
            truth_box = self.dtype(np.zeros((n, 4)))
            truth_box[:n, 2] = truth_w_all[b, :n]
            truth_box[:n, 3] = truth_h_all[b, :n]
            truth_i = truth_i_all[b, :n]
            truth_j = truth_j_all[b, :n]
            truth_box[:n, 0] = truth_x_all[b, :n]
            truth_box[:n, 1] = truth_y_all[b, :n]
            
            # calculate iou between truth and reference anchors
            best_n_all,best_n_mask = self.get_best_mask(truth_box)
            best_n = best_n_all % 3

            pred_best_iou = self.get_best_iou(pred[b],truth_box)
            # set mask to zero (ignore) if pred matches truth
            obj_mask[b] = 1 - pred_best_iou.type(self.dtype)

            if sum(best_n_mask) == 0:
                continue

            for ti in range(best_n.shape[0]):
                if best_n_mask[ti] == 1:
                    i, j = truth_i[ti], truth_j[ti]
                    a = best_n[ti]
                    obj_mask[b, a, j, i] = 1
                    tgt_mask[b, a, j, i, :] = 1
                    target[b, a, j, i, 0] = truth_x_all[b, ti] - truth_x_all[b, ti].to(torch.int16).to(torch.float)
                    target[b, a, j, i, 1] = truth_y_all[b, ti] - truth_y_all[b, ti].to(torch.int16).to(torch.float)
                    target[b, a, j, i, 2] = torch.log(truth_w_all[b, ti] / torch.Tensor(self.masked_anchors)[best_n[ti], 0] + 1e-16)
                    target[b, a, j, i, 3] = torch.log(truth_h_all[b, ti] / torch.Tensor(self.masked_anchors)[best_n[ti], 1] + 1e-16)
                    target[b, a, j, i, 4] = 1
                    target[b, a, j, i, 5 + labels[b, ti,0].to(torch.int16).numpy()] = 1
                    tgt_scale[b, a, j, i, :] = torch.sqrt(2 - truth_w_all[b, ti] * truth_h_all[b, ti] / fsize / fsize)

        # loss calculation
        return self.get_losses(output,target,obj_mask,tgt_mask,tgt_scale)
