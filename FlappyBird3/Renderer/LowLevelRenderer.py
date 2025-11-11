import pygame

class PrimitiveRenderer:
    def __init__(self, width, height,bird_radius=13): 
        self.width = width
        self.height = height
        self.clock = pygame.time.Clock()
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.bird_radius = bird_radius
        pygame.init()


    def render(self,bird,pipes):
        self._screen_renderer()
        self._pipes_renderer(pipes)
        self._bird_renderer(bird)
        pygame.display.flip()
        self.clock.tick(30)    

    def _screen_renderer(self):
        self.screen.fill((135,206,250))
    
    def _bird_renderer(self,bird,color=(255,140,0)):
        pygame.draw.circle(self.screen, color,(50,int(bird.y)),self.bird_radius)

    def _pipes_renderer(self,pipes):
        for num in range(len(pipes)):
            self._pipe_renderer(pipes[num])    

    def _pipe_renderer(self,pipe,color=(34, 139, 34)):
        pygame.draw.rect(self.screen, color, (pipe.x, 0, pipe.thick, pipe.gap_center - pipe.gap))
        pygame.draw.rect(self.screen, color, (pipe.x, pipe.gap_center + pipe.gap, pipe.thick, self.height))     

    def textRenderer(self,text,Font="ArialBlack",font_size=32,wait=900,color=(255,0,0),**kwargs):
        font = pygame.font.SysFont(Font,font_size)
        text = font.render(text,True,color)
        rect = text.get_rect(**kwargs)
        self.screen.blit(text,rect)
        pygame.display.flip()
        pygame.time.wait(wait) 