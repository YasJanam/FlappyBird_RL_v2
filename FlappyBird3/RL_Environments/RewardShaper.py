
class reward_shaper:
    def __init__(self):
        pass
    def BaseReward(self):
        pass
    def GuidanceReward(self):
        pass
    def ProgressReward(self):
        pass
    def TerminatePenalty(self):
        pass

class RewardShaper(reward_shaper):
    def __init__(self):
        super().__init__()

    def BaseReward(self):
        return 0.05
    
    def GuidanceReward(self,env):
        distance_to_center = abs(env.bird.y - env.pipes_manager.pipes[0].gap_center)
        distance_to_center = distance_to_center / (env.screen_height / 3)
        return 0.7 * (1 - distance_to_center) 
    
    def SmoothnessReward(self,env):
        return -0.05 * abs(env.bird.velocity)
    
    def ProgressReward(self):
        return 5 
    
    def TerminatePenalty(self):
        return -5 
    
    def reward(self,env):
        rew = self.BaseReward()
        rew += self.SmoothnessReward(env)
        rew += self.GuidanceReward(env)
        if env.pipes_manager.pipe_exited_screen():rew += self.ProgressReward()
        if env.terminated: rew = self.TerminatePenalty()  
        return rew

