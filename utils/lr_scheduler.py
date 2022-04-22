import torch.optim as optim

class LambdaLRScheduler():
    def __init__(self,optimizer,cfg,iter_state=0):
        self.iter_state = iter_state
        self.cfg=cfg
        self.iter_size = self.cfg['SOLVER']['MAXITER']
        self.batch_size = self.cfg['DATA']['BATCHSIZE']
        self.subdivision = self.cfg['DATA']['SUBDIVISION']
        self.effective_batch_size = self.batch_size*self.subdivision
        burn_in = self.cfg['SOLVER']['BURN_IN']
        steps = eval(self.cfg['SOLVER']['STEPS'])

        # Learning rate setup
        def burnin_schedule(i):
            if i < burn_in:
                return pow(i / burn_in, 4)
            elif i < steps[0]:
                return 1.0
            elif i < steps[1]:
                return 0.1
            else:
                return 0.01
        
        self.scheduler = optim.lr_scheduler.LambdaLR(optimizer,burnin_schedule)
    
    def get_current_lr(self):
        return self.scheduler.get_last_lr()[0] *self.effective_batch_size
    
    def load_state_dict(self,state):
        self.scheduler.load_state_dict(state['scheduler_state_dict'])
    
    def state_dict(self):
        return self.scheduler.state_dict()
    
    def step(self):
        self.scheduler.step()
        self.iter_state = self.iter_state+1