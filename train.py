from __future__ import division

from models.yolov3 import YOLOv3
from dataset.cocodataset import COCODataset
from utils.misc import parse_args,iscuda,load_cfg
from utils.ckpt_utils import load_ckpt,save_ckpt
from utils.logging import init_wandb
from utils.cocoapi_evaluator import COCOAPIEvaluator

import os
try:
    import wandb 
except ModuleNotFoundError:
    pass

import torch
import torch.optim as optim


def main(args):
    """
    YOLOv3 trainer. See README for details.
    """
    
    # Log onto WandB
    if args.wandb_API_key is not None:
        os.environ['WANDB_API_KEY']=args.wandb_API_key
    # Parse config settings
    cfg = load_cfg(args.cfg)
    
    batch_size = cfg['DATA']['BATCHSIZE']
    subdivision = cfg['DATA']['SUBDIVISION']
    effective_batch_size = batch_size*subdivision
    print('effective_batch_size = batch_size * iter_size = %d * %d= %d' %(batch_size, subdivision,effective_batch_size))
    
    dataset = COCODataset(cfg,mode='train',debug=cfg['DEBUG'])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True,num_workers=cfg['NUM_CPUS'])
    evaluator = COCOAPIEvaluator(cfg)

    # Initiate model
    model = YOLOv3(cfg)
    
    # optimizer setup
    optimizer = optim.SGD(
                           model.get_params(), 
                           lr=cfg['SOLVER']['LR'] / batch_size / subdivision, 
                           momentum=cfg['SOLVER']['MOMENTUM'],
                           dampening=0, 
                           weight_decay=cfg['SOLVER']['DECAY']*effective_batch_size
                         )

    # Learning rate setup
    def burnin_schedule(i):
        burn_in = cfg['SOLVER']['BURN_IN']
        steps = eval(cfg['SOLVER']['STEPS'])
        if i < burn_in:
            factor = pow(i / burn_in, 4)
        elif i < steps[0]:
            factor = 1.0
        elif i < steps[1]:
            factor = 0.1
        else:
            factor = 0.01
        return factor
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, burnin_schedule)

    if cfg["LOGGING"]["TYPE"].upper() == "WANDB":
        init_wandb(cfg)
    if args.wandb_checkpoint is not None:
        wandb_checkpoint = wandb.restore(os.path.join(ckpt_dir,args.wandb_checkpoint))
        ckpt_path = wandb_checkpoint.name
    else:
        ckpt_path = args.checkpoint
    model,optimizer,scheduler,iter_state = load_ckpt(ckpt_path,model,optimizer,scheduler)

    cuda = iscuda(cfg)
    if cuda: 
        model = model.cuda()

    model.train()

    dataiterator = iter(dataloader)
    iter_size = cfg['SOLVER']['MAXITER']
    # start training loop
    for iter_i in range(iter_state, iter_size + 1):

        # subdivision loop
        optimizer.zero_grad()
        for inner_iter_i in range(subdivision):
            try:
                imgs, targets, _, _ = next(dataiterator)  # load a batch
            except StopIteration:
                dataiterator = iter(dataloader)
                imgs, targets, _, _ = next(dataiterator)  # load a batch
            loss = model(imgs, targets,cuda=cuda)
            if cfg["LOGGING"]["TYPE"].upper() == "WANDB":
                wandb.watch(model,criterion=loss,log="all")
            loss.backward()

        optimizer.step()
        scheduler.step()

        if iter_i % 10 == 0:
            # logging
            current_lr = scheduler.get_last_lr()[0] * batch_size * subdivision
            # print('[Iter %d/%d] [lr %f] '
            #       '[Losses: xy %f, wh %f, conf %f, cls %f, l2 %f]'
            #       % (iter_i, iter_size, current_lr,
            #          model.loss_dict['xy'], model.loss_dict['wh'],
            #          model.loss_dict['conf'], model.loss_dict['cls'], 
            #          model.loss_dict['l2']),
            #       flush=True)
            print('[Iter %d/%d] [lr %f][Loss %f] '% (iter_i, iter_size, current_lr,loss.item()),flush=True)
            if cfg["LOGGING"]["TYPE"].upper() == "WANDB":
                train_loss = {
                                'XY loss': model.loss_dict['xy'],
                                'WH loss' : model.loss_dict['wh'],
                                'Conf loss' : model.loss_dict['conf'],
                                'Cls loss' : model.loss_dict['cls'],
                             }
                wandb.log({
                            'Sub_Loss': train_loss,
                            'L2_Loss' : model.loss_dict['l2'],
                            'Total Loss' : loss.item(), 
                            'Learning Rate': current_lr,
                          }, step=iter_i)

            # random resizing
            if cfg['AUGMENTATION']['RANDRESIZE']:
                dataset.random_resize()
                dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True,num_workers=cfg['NUM_CPUS'])
                dataiterator = iter(dataloader)
        
        # COCO evaluation
        # if iter_i % cfg["TEST"]["EVAL_INTERVAL"] == 0 and iter_i > 0:
        #     ap50_95,ap50 = evaluator.evaluate(model)
        #     model.train()
        #     if cfg["LOGGING"]["TYPE"].upper() == "WANDB":
        #         wandb.log({
        #                     'val/COCOAP50': ap50,
        #                     'val/COCOAP50_95' : ap50_95
        #                   }, step=iter_i)

        # save checkpoint
        if iter_i > 0 and (iter_i % cfg["SAVING"]["CKPT_INTERVAL"] == 0):
            save_ckpt(cfg["SAVING"]["CKPT_DIR"],iter_i,model,optimizer,scheduler)
            if cfg["LOGGING"]["TYPE"].upper() == "WANDB":
                wandb.save(os.path.join(ckpt_dir, "YOLO"+str(it)+".ckpt"))
    if cfg["LOGGING"]["TYPE"].upper() == "WANDB":
        wandb.finish()


if __name__ == '__main__':
    args = parse_args()
    main(args)
