import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

class EpisodicActorCriticTrainer:
    def __init__(self, gamma=0.99, actor=None, critic=None, actor_lr=5e-4, critic_lr=4e-4):
        #self.env = env
        self.actor = actor
        self.critic = critic

        self.actor_lr = actor_lr
        self.critic_lr = critic_lr

        self.actor_opt = optim.Adam(actor.parameters(), actor_lr)
        self.critic_opt = optim.Adam(critic.parameters(), critic_lr)

        self.gamma = gamma
 

    def train(self, env,num_episodes,log_interval=100):
        self.actor.train()
        self.critic.train()
        for episode in range(num_episodes):
            obs, _ = env.reset()
            obs = torch.as_tensor(obs, dtype=torch.float32)
            episode_reward = 0
            done = False

            rewards = []
            log_probs = []
            values = []

            while not done:               
                action, log_prob = self.actor.act(obs)
                next_obs, reward, term, trunc,_ = env.step(action) 
                next_obs = torch.as_tensor(next_obs, dtype=torch.float32)

                value = self.critic(obs)
                done = term or trunc

                log_probs.append(log_prob)
                rewards.append(reward)
                values.append(value)

                obs = next_obs
            values = torch.cat(values).squeeze(-1)

            # compute returns
            returns, G = [], 0
            for r in reversed(rewards):
                G = r + self.gamma * G
                returns.insert(0,G)
            returns = torch.tensor(returns)
            #returns = (returns - returns.mean()) / (returns.std() + 1e-8)

            # advantage
            advantages = returns - values.detach()
            #advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            #advantages = advantages.clamp(-10,10)

            # update actor
            self.actor_opt.zero_grad()
            actor_loss = -(torch.stack(log_probs) * advantages).mean()
            actor_loss.backward()
            self.actor_opt.step()

            # update critic
            self.critic_opt.zero_grad()
            critic_loss = (returns - values).pow(2).mean()
            critic_loss.backward()
            #nn.utils.clip_grad_norm_(self.critic.parameters(),0.4)
            self.critic_opt.step()            

            # log
            episode_reward = sum(rewards)
            if episode % log_interval == 0 or episode == (num_episodes - 1):
                print(f"Episode: {episode},  Reward: {episode_reward:.2f}")
        env.close()  

    def oneTest(self,env):
        state, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32)
            with torch.no_grad():
                action, log_prob = self.actor.act(state_tensor)  
            next_state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
            state = next_state                
        return total_reward

    def test(self,env,num_tests):
        total_rewards = []
        for _ in range(num_tests):
            total_rewards.append(self.oneTest(env))
        rewards = [float(x) for x in total_rewards]
        return rewards              

