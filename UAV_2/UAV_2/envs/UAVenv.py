
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import logging
log = logging.getLogger(__name__)
log.info('%s logger started.',__name__)

class UAVenv(gym.Env):
  
    metadata = {'render.modes': ['human']}
    def __init__(self):
#         super(UAVenv, self).__init__()
        self.action_space = spaces.Box(\
            low=np.array([-.1]), high=np.array([.1]), dtype=np.float16)
        self.observation_space = spaces.Box(\
            low=np.array([-5,0,0]), high=np.array([5,100,10]), dtype=np.float16)
        self.state=[]
        self.count=0
        self.reward = 0
        self.done = False
        self.des = np.array([100,5])
        
    
    def step(self, action):
        self.count+=1
        done =False
        v = np.squeeze(action)
        old_state = self.state
        new_state = np.zeros_like(self.state)
        new_state[0]=self.state[0]+v
        new_state[1] = self.state[1]+np.cos(new_state[0]/180*np.pi)*2
        new_state[2] = self.state[2]+np.sin(new_state[0]/180*np.pi)*2
        old_dis = np.linalg.norm(self.des-old_state[1:])
        new_dis = np.linalg.norm(self.des-new_state[1:])
        self.reward = old_dis-new_dis
        self.state = new_state
        if self.state[-2]>self.des[-2]:
            done =True
            if np.linalg.norm(self.des-self.state[1:])<2:
                self.reward = 100
            else:
                self.reward = -100
        return self.state,self.reward,done,{}
    def reset(self):
        self.count=0
        self.state = np.array([0,0,0])
        self.v0 = self.state[0]
        return self.state
        
    def render(self, mode='human', close=False):
        ### this one we just leave it like this.
        pass  