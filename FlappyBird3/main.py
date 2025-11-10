from Objects.Bird import Bird
from Objects.Pipe import PipesManagement
from Renderer.LowLevelRenderer import PrimitiveRenderer
from RL_Environments.Env import MultiPipeEnv
from RL_Environments.DoneManager import DoneManager
from RL_Environments.RewardShaper import RewardShaper
from RLTrainer.ActorCritic import EpisodicActorCriticTrainer
from Value_Policy_Networks.Actor import DiscreteActor
from Value_Policy_Networks.Critic import ContinuousCritic
import pygame, gc
import time


# --- define ---
# arguments 
screen_width=800
screen_height=512

flap_power = -5.0    #-3.5
bird_init_y = screen_height // 2
bird_init_vel = 0

pipe_speed = 2.6
pipe_height = screen_height
pipe_init_x = screen_width
pipe_thick=64
min_gap=40
max_gap=60

gravity = 2.5
max_steps = 600  # for truncating
min_pipes_spacing = 160 # spacing between pipes
max_pipes_spacing = 280
max_pipes = 6 # max number of pipes in screen

# objects :
bird = Bird(flap_power,bird_init_vel,bird_init_y)
reward_shaper = RewardShaper()
done_manager = DoneManager()
pipe_manager = PipesManagement(pipe_speed,pipe_height,pipe_init_x,min_gap,
                               max_gap,pipe_thick,min_pipes_spacing,max_pipes_spacing,max_pipes)


# Environment 
env = MultiPipeEnv(gravity,screen_height,screen_width,max_steps,bird,pipe_manager,
                   Reward_Shaper=reward_shaper,Done_Manager=done_manager)
       
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
env2 = MultiPipeEnv(gravity,screen_height,screen_width,max_step,bird,pipe_manager,renderer,
                    render_mode=True,Reward_Shaper=reward_shaper,Done_Manager=done_manager)

# test
trainer.test(env2,4)
       
       