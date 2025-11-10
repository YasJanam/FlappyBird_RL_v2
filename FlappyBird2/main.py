from Objects.Bird import Bird
from Objects.Pipe import Pipe
from Renderer.LowLevelRenderer import PrimitiveRenderer
from RL_Environments.DecoupledEnv import SinglePipeEnv
from RLTrainer.ActorCritic import EpisodicActorCriticTrainer
from Value_Policy_Networks.Actor import DiscreteActor
from Value_Policy_Networks.Critic import ContinuousCritic
import torch
import pygame, gc
import time


# --- define ---
# screen
screen_width=300
screen_height=512


# -- Bird --
flap_power = -4     #-3.5
bird_init_y = screen_height // 2
bird_init_vel = 0

# -- Pipe --
pipe_speed = 2.7
pipe_height = screen_height
pipe_init_x = screen_width
pipe_thick=80
min_gap=40
max_gap=60

gravity = 2.5
max_steps = 600  # for truncating

# objects :
bird = Bird(flap_power,bird_init_vel,bird_init_y)
pipe = Pipe(pipe_speed,pipe_height,pipe_init_x,min_gap,max_gap,pipe_thick)

# Environment 
env = SinglePipeEnv(gravity,screen_height,screen_width,max_steps,bird,pipe)
       
hidden_dim = 64
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.n

actor = DiscreteActor(hidden_dim=hidden_dim ,obs_dim=obs_dim, act_dim=act_dim)
critic = ContinuousCritic(hidden_dim=hidden_dim, obs_dim=obs_dim)
trainer = EpisodicActorCriticTrainer(gamma=0.99, actor=actor, critic=critic, actor_lr=5e-4, critic_lr=4e-4)

num_episodes = 2000
trainer.train(env,num_episodes,log_interval=100)

pygame.quit()
pygame.display.quit()
gc.collect()
time.sleep(0.5)

# Environment & Renderer
max_step = float('inf')
renderer = PrimitiveRenderer(screen_width, screen_height)
env2 = SinglePipeEnv(gravity,screen_height,screen_width,max_step,bird,pipe,renderer,render_mode=True)

# test
trainer.test(env2,3)
       