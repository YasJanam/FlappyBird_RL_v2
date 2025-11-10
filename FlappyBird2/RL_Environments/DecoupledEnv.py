import gymnasium as gym
from gymnasium import spaces
import numpy as np

class SinglePipeEnv(gym.Env):
    def __init__(self,gravity,screen_height,screen_width,max_steps=700,bird=None,pipe=None,
                 renderer=None,render_mode=False):
        super(SinglePipeEnv,self).__init__()
        self.gravity = gravity
        self.screen_height = screen_height
        self.screen_width = screen_width

        self.bird = bird
        self.pipe = pipe

        self.render_mode = render_mode        
        self.renderer = renderer

        self.max_steps = max_steps
        self.steps = 0

        self.reward = 0

        # gym spaces
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)

        self.reset()


    def step(self,action):
        self.steps += 1

        # bird and pipe 
        self.bird.flap() if action == 1 else self.bird.FreeFall(self.gravity)  
        self.pipe.move()
        
        # reward and done
        terminated, truncated = self.Done()      
        if self.done and self.render_mode: self.renderer.textRenderer(text="Done",center=(self.screen_width//2,self.screen_height//2))
        self.Reward()

   
        # --- pipe ---
        if self.pipe.pipe_exited_screen(): self.pipe.reset(self.screen_width)

        self.render()

        obs = np.array([self.bird.y, self.bird.velocity, self.pipe.x, self.pipe.gap_center],dtype=np.float32)
        return obs, self.reward, terminated, truncated, {}   
       
    def Reward(self):
        self.reward = 0.05  # Base Reward  
        self.reward += -0.05 * abs(self.bird.velocity) # smoothness reward

        distance_to_center = abs(self.bird.y - self.pipe.gap_center)
        distance_to_center = distance_to_center / (self.screen_height / 3)
        self.reward += 0.7 * (1 - distance_to_center)   # guidance reward  

        if self.pipe.pipe_exited_screen() : self.reward += 5   # progress reward
        if self.Terminated() : self.reward = -5   
  
    def bird_pipe_collision(self):
        return abs(self.bird.y - self.pipe.gap_center) >= (self.pipe.gap / 2)  and 0 <= self.pipe.x <= (self.pipe.thick / 2)
    
    def bird_OutOf_screen(self):
        return self.bird.y < 0 or self.bird.y > self.screen_height
    
    def Done(self):
        truncated = self.Truncated()
        terminated = self.Terminated()
        self.done = truncated or terminated
        return terminated,truncated 
    
    def Truncated(self):
       return self.steps >= self.max_steps
    
    def Terminated(self):
        terminated= True if self.bird_pipe_collision() or self.bird_OutOf_screen() else False     
        return terminated

    def reset(self):
        self.steps = 0
        self.reward = 0
        self.done = False
        self.bird.reset()
        self.pipe.reset(self.screen_width)
        obs = np.array([self.bird.y,self.bird.velocity,self.pipe.x,self.pipe.gap_center],dtype=np.float32)
        return obs,{} 

    def render(self):
        if self.render_mode:
            self.renderer.render(self.bird,self.pipe)
