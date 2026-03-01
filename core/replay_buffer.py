import random
import numpy as np

class ReplayBuffer:

    def __init__(self,capacity):
        self.capacity=capacity
        self.data=[]
        self.pos=0

    def add(self,*transition):
        if len(self.data)<self.capacity:
            self.data.append(transition)
        else:
            self.data[self.pos]=transition
        self.pos=(self.pos+1)%self.capacity

    def sample(self,batch):
        batch=random.sample(self.data,batch)
        s,a,r,s2,d=map(np.array,zip(*batch))
        return s,a,r,s2,d
