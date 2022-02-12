import torch
import os 

def save_ckpt(ckpt_dir,it,model,optimizer,scheduler):
    save_dict = {'iter': it,
                 'model_state_dict': model.state_dict(),
                 'optimizer_state_dict': optimizer.state_dict(),
                 'scheduler_state_dict' : scheduler.state_dict()
                }
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "YOLO"+str(it)+".ckpt")
    torch.save(save_dict,ckpt_path)

def load_ckpt(ckpt_path,model,optimizer,scheduler):
    iter_state = 0
    if ckpt_path is not None:
        print("loading pytorch ckpt...",ckpt_path)
        state = torch.load(ckpt_path)
        model.load_state(state)
        if 'optimizer_state_dict' in state.keys():
            optimizer.load_state_dict(state['optimizer_state_dict'])
            iter_state = state['iter'] + 1
        if 'scheduler_state_dict' in state.keys():
            scheduler.load_state_dict(state['scheduler_state_dict'])
    return model,optimizer,scheduler,iter_state
