from collections import defaultdict

class Loss_Dict():
    def __init__(self):
        self.dict = defaultdict(float)
        self.name_list = []
        self.count = 0

    def init(self,name_list):
        if name_list is not None:
            self.name_list =  name_list
            for name in name_list:
                self.dict[name] = 0
    
    def update(self,name_list=None,losses=None):
        if name_list is not None:
            assert len(name_list) == len(losses)
            self.name_list = name_list
        for name,loss in zip(self.name_list,losses):
            self.dict[name]+=loss
        self.count= self.count+1

    def reset(self,name_list=None):
        if name_list is not None:
            self.name_list = name_list
        for name in self.name_list:
            self.dict[name]=0
        self.count = 0
    
    def freeze(self):
        if self.count!=0:
            for name in self.dict.keys():
                self.dict[name] = self.dict[name]/self.count


