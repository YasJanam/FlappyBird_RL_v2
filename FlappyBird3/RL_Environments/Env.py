import gymnasium as gym
from gymnasium import spaces
import numpy as np
from .RewardShaper import RewardShaper
from .DoneManager import DoneManager

class MultiPipeEnv(gym.Env):
    def __init__(self,gravity,screen_height,screen_width,max_steps=700,bird=None,pipes_manager=None,
                 renderer=None,render_mode=False,Reward_Shaper=None,Done_Manager=None):
        super(MultiPipeEnv,self).__init__()
        self.gravity = gravity
        self.screen_height = screen_height
        self.screen_width = screen_width

        self.bird = bird
        self.pipes_manager = pipes_manager

        self.render_mode = render_mode        
        self.renderer = renderer

        self.max_steps = max_steps
        self.steps = 0

        self.reward = 0
        self.reward_shaper = Reward_Shaper 
        if RewardShaper is None: 
            self.reward_shaper = RewardShaper()

        self.done_manager = Done_Manager
        if DoneManager is None: 
            self.done_manager = DoneManager()

        # gym spaces
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)

        self.reset()


    def step(self,action):
        self.steps += 1

        # bird and pipe 
        self.bird.update(action,self.gravity) 
        self.pipes_manager.move()
        
        # reward and done
        self.terminated, self.truncated = self.done_manager.Done(self)    
        self.done = self.terminated or self.truncated  

        if self.done and self.render_mode: 
            self.renderer.textRenderer(text="Done",center=(self.screen_width//2,self.screen_height//2))

        self.reward = self.reward_shaper.reward(self)
        self.pipes_manager.update()

        self.render()

        obs = np.array([self.bird.y, self.bird.velocity, self.pipes_manager.pipes[0].x,self.pipes_manager.pipes[0].gap_center],dtype=np.float32)
        return obs, self.reward, self.terminated, self.truncated, {}   
   
    def reset(self):
        self.steps = 0
        self.reward = 0
        self.terminated = False
        self.truncated = False
        self.done = False
        self.bird.reset()
        self.pipes_manager.reset()
        obs = np.array([self.bird.y,self.bird.velocity,self.pipes_manager.pipes[0].x,self.pipes_manager.pipes[0].gap_center],dtype=np.float32)
        return obs,{} 

    def render(self):
        if self.render_mode:
            self.renderer.render(self.bird,self.pipes_manager.pipes)
