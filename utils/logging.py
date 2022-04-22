import os 
try:
    import wandb 
except ModuleNotFoundError:
    pass

def init_wandb(cfg) -> None:
    """
    Initialize project on Weights & Biases
    Args:
        cfg   (Dict) : Configuration file
    """
    wandb.init(
        name=cfg["LOGGING"]["NAME"],
        config=cfg,
        project=cfg["LOGGING"]["PROJECT"],
        resume="allow",
        id=cfg["LOGGING"]["ID"]
    )

def restore_ckpt(wandb_ckpt_path):
    wandb_checkpoint = wandb.restore(wandb_ckpt_path)
    return wandb_checkpoint.name

def wandb_log(loss_dict,current_lr,iter_i):
    train_loss = {
        'XY loss':loss_dict['xy'],
        'WH loss' :loss_dict['wh'],
        'Conf loss' :loss_dict['conf'],
        'Cls loss' :loss_dict['cls'],}
    wandb.log({
        'Sub_Loss': train_loss,
        'L2_Loss' : loss_dict['l2'],
        'Total Loss' :loss_dict['total'], 
        'Learning Rate': current_lr,
        }, step=iter_i)

def log(scheduler,model):
    model.loss_dict.freeze()
    loss = model.loss_dict.dict['total']
    current_lr = scheduler.get_current_lr()
    iter_i = scheduler.iter_state
    iter_size = scheduler.iter_size
    print('[Iter %d/%d] [lr %f][Loss %f] '% (iter_i, iter_size, current_lr,loss),flush=True)
    if cfg["LOGGING"]["TYPE"].upper() == "WANDB":
        wandb_log(model.loss_dict.dict,current_lr,iter_i)
    model.loss_dict.reset()