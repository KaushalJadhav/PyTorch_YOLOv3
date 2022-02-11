import json
import tempfile

from pycocotools.cocoeval import COCOeval
from torch.autograd import Variable

from dataset.cocodataset import *
from utils.utils import *
from utils.aug_utils import *


class COCOAPIEvaluator():
    """
    COCO AP Evaluation class.
    All the data in the val2017 dataset are processed \
    and evaluated by COCO API.
    """
    def __init__(self,cfg):
        """
        Args:
            cfg (dict) : Configuration file.
            confthre (float):
                confidence threshold ranging from 0 to 1, \
                which is defined in the config file.
            nmsthre (float):
                IoU threshold of non-max supression ranging from 0 to 1.
        """

        self.dataset = COCODataset(cfg,mode='val')
        self.dataloader = torch.utils.data.DataLoader(self.dataset,batch_size=1, shuffle=False, num_workers=0)
        self.confthre = cfg['TEST']['CONFTHRE'] # from darknet
        self.nmsthre = cfg['TEST']['NMSTHRE'] # 0.45 (darknet)

        cuda = torch.cuda.is_available()
        self.dtype = torch.cuda.FloatTensor if cuda else torch.FloatTensor

        self.coco = self.dataset.coco
        self.dataiterator = iter(self.dataloader)
        self.annType = ['segm', 'bbox', 'keypoints']
    
    def get_output(self,img,model):
        outputs = model(img)
        outputs = postprocess(outputs, 80, self.confthre, self.nmsthre)
        if outputs[0] is None:
            return None
        return outputs[0].cpu().data
    
    def get_score(self,output,info_img):
        for output in outputs:
                x1 = float(output[0])
                y1 = float(output[1])
                x2 = float(output[2])
                y2 = float(output[3])
                label = self.dataset.class_ids[int(output[6])]
                box = yolobox2label((y1, x1, y2, x2), info_img)
                bbox = [box[1], box[0], box[3] - box[1], box[2] - box[0]]
                score = float(output[4].data.item() * output[5].data.item()) # object score * class score
                return {"image_id": id_, "category_id": label, "bbox": bbox,
                     "score": score, "segmentation": []} # COCO json format

    def evaluate(self, model):
        """
        COCO average precision (AP) Evaluation. Iterate inference on the test dataset
        and the results are evaluated by COCO API.
        Args:
            model : model object
        Returns:
            ap50_95 (float) : calculated COCO AP for IoU=50:95
            ap50 (float) : calculated COCO AP for IoU=50
        """
        model.eval()
        
        ids = []
        data_dict = []

        while True: # all the data in val2017
            try:
                img, _, info_img, id_ = next(self.dataiterator)  # load a batch
            except StopIteration:
                break
            info_img = [float(info) for info in info_img]
            id_ = int(id_)
            ids.append(id_)
            with torch.no_grad():
                img = Variable(img.type(self.dtype))
                outputs = self.get_output(img,model)
                if outputs[0] is None:
                    continue 
            data_dict.append(self.get_score(output,info_img))

        # Evaluate the Dt (detection) json comparing with the ground truth
        return self.coco_eval(data_dict,ids)
        
    def coco_eval(self,data_dict,ids):
        if len(data_dict) > 0:

            # workaround: temporarily write data to json file because pycocotools can't process dict in py36.
            _, tmp = tempfile.mkstemp()
            json.dump(data_dict, open(tmp, 'w'))
            cocoDt = self.coco.loadRes(tmp)
            cocoEval = COCOeval(self.dataset.coco,cocoDt,self.annType[1])
            cocoEval.params.imgIds = ids
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()
            return cocoEval.stats[0], cocoEval.stats[1]
        else:
            return 0, 0

