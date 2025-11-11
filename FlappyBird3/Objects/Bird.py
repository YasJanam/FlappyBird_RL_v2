
class bird:
    def __init__(self):
        pass
    def reset(self):
        pass
    def flap(self):
        pass

class Bird(bird):
    def __init__(self,flap_power=3,init_vel=0,init_y=144):
        super().__init__() 
        self.y = init_y
        self.velocity = init_vel
        self.flap_power = flap_power       
        self.init_vel = init_vel  # initial velocity
        self.init_y = init_y

    def reset(self):
        self.y = self.init_y
        self.velocity = self.init_vel  

    def flap(self):
        self.velocity = self.flap_power
        self.y += self.velocity   

    def FreeFall(self,gravity):    
        self.velocity += gravity
        self.y += self.velocity   

    def update(self,action,gravity):
        self.flap() if action == 1 else self.FreeFall(gravity)  
        