
import random

class pipe:
    def __init__(self):
        pass
    def reset(self):
        pass
    def move(self):
        pass

class   Pipe(pipe):
    def __init__(self,speed=3,pipe_height=512,init_x=288,min_gap=40,max_gap=60,pipe_thick=30):
        super().__init__()   
        self.x = init_x
        self.speed = speed
        self.gap = None
        self.gap_center = None
        self.init_x = init_x
        self.pipe_height = pipe_height 
        self.min_gap = min_gap
        self.max_gap = max_gap
        self.thick = pipe_thick


    def reset(self,screen_width):
        self.x = screen_width

        # reset gap-size
        self.reset_gap(self.min_gap,self.max_gap)

        # reset gap-center
        min_gap_center = (self.min_gap // 2) + 10 
        max_gap_center = self.pipe_height - min_gap_center
        self.reset_gap_center(min_gap_center,max_gap_center)

    def reset_gap(self,min_gap_size,max_gap_size):
        self.gap = random.randint(min_gap_size,max_gap_size)

    def reset_gap_center(self,min_gap_center,max_gap_center):
        self.gap_center = random.randint(min_gap_center ,max_gap_center)  

    def move(self):
        self.x -= self.speed  

    def pipe_exited_screen(self):
        return self.x < -0.1


class PipesManagement(pipe):
    def __init__(self,speed=3,pipe_height=512,screen_width=288,min_gap=40,
                 max_gap=60,pipe_thick=30,min_pipes_spacing=150,max_pipes_spacing=300,max_pipes=float('inf')):
        super().__init__()
        self.pipes = []
        self.screen_width = screen_width
        self.speed = speed
        self.pipe_height = pipe_height 
        self.min_gap = min_gap
        self.max_gap = max_gap
        self.thick = pipe_thick
        #self.pipes_spacing = pipes_spacing  # spacing between pipes
        self.min_pipes_spacing = min_pipes_spacing
        self.max_pipes_spacing = max_pipes_spacing
        self.max_pipes = max_pipes  # max number of pipes in screen

    def move(self): # move all pipes
        for pipe in self.pipes:
            pipe.move()

    def pipe_exited_screen(self):
        return self.pipes[0].pipe_exited_screen()

    def _add_pipe(self):
        if len(self.pipes) < self.max_pipes:
            new_pipe = Pipe(self.speed,self.pipe_height,self.screen_width,self.min_gap,self.max_gap,self.thick) 
            new_pipe.reset(self.screen_width)
            self.pipes.append(new_pipe)

    def _remove_pipe(self):
        if len(self.pipes) > 0:
            self.pipes.pop(0)

    def _can_add_pipe(self):
        pipes_spacing = random.randint(self.min_pipes_spacing,self.max_pipes_spacing)
        dis = self.pipes[-1].x + (self.pipes[-1].thick) + pipes_spacing  
        return True if dis < self.screen_width else False
    
    
    def update(self):
        #self.move() # move all pipes     
        if self._can_add_pipe():
            self._add_pipe()

        if self.pipe_exited_screen():
            self._remove_pipe()
            
    def reset(self):
        self.pipes = []
        new_pipe = Pipe(self.speed,self.pipe_height,self.screen_width,self.min_gap,self.max_gap,self.thick) 
        new_pipe.reset(self.screen_width)
        self.pipes.append(new_pipe) 



