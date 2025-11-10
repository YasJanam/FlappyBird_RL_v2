
class done_manager:
    def __init__(self):
        pass
    def BirdPipeCollision(self):
        pass
    def BirdOutOfScreen(self):
        pass
    def Terminate(self):
        pass
    def Truncate(self):
        pass
    def Done(self):
        pass

class DoneManager(done_manager):
    def __init__(self):
        super().__init__()

    def BirdPipeCollision(self,env):
        return abs(env.bird.y - env.pipes_manager.pipes[0].gap_center) >= (env.pipes_manager.pipes[0].gap / 2) \
            and 0 <= env.pipes_manager.pipes[0].x <= (env.pipes_manager.pipes[0].thick / 2)
    
    def BirdOutOfScreen(self,env):
        return env.bird.y < 0 or env.bird.y > env.screen_height
    
    def Truncate(self,env):
        return env.steps >= env.max_steps
    
    def Terminate(self,env):
        terminated= True if self.BirdPipeCollision(env) or self.BirdOutOfScreen(env) else False 
        return terminated
    
    def Done(self,env):
        truncated = self.Truncate(env)
        terminated = self.Terminate(env)
        return terminated,truncated  