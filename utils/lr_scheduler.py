import torch.optim as optim

class LambdaLRScheduler():
    def __init__(self,optimizer,cfg,iter_state=0):
        self.scheduler = optim.lr_scheduler.LambdaLR(optimizer,self.burnin_schedule)
        self.iter_state = iter_state
        self.cfg=cfg
        self.iter_size = self.cfg['SOLVER']['MAXITER']
        self.batch_size = self.cfg['DATA']['BATCHSIZE']
        self.subdivision = self.cfg['DATA']['SUBDIVISION']
        self.effective_batch_size = self.batch_size*self.subdivision
        self.burn_in = self.cfg['SOLVER']['BURN_IN']
        self.steps = eval(self.cfg['SOLVER']['STEPS'])
    
    # Learning rate setup
    def burnin_schedule(self,i):
        if i < self.burn_in:
            factor = pow(i / self.burn_in, 4)
        elif i < self.steps[0]:
            factor = 1.0
        elif i < self.steps[1]:
            factor = 0.1
        else:
            factor = 0.01
        return factor
    
    def get_current_lr(self):
        return self.scheduler.get_last_lr()[0] *self.effective_batch_size
    
    def load_state_dict(self,state):
        self.scheduler.load_state_dict(state['scheduler_state_dict'])
    
    def state_dict(self):
        return self.scheduler.state_dict()
    
    def step(self):
        self.scheduler.step()
        self.iter_state = self.iter_state+1