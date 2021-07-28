import pygame
import pickle


class ObstacleCreator:
    def __init__(self, screen):
        self.screen = screen
        self.win_size = [640,480]
        self.clicked_rect = False
        self.obstacle_list = pygame.sprite.Group()
        self.angle = 0
        self.standard_size = [90,20]
        self.size = self.standard_size
        # Text representation
        self.font_size = 12
        self.font = pygame.font.Font('freesansbold.ttf', self.font_size)

    def draw_selector(self):
        self.rect = pygame.draw.rect(self.screen, (0,255,0), (self.win_size[0]-98,140,self.standard_size[0],self.standard_size[1]))
        self.finish_creator = pygame.draw.rect(self.screen, (0,255,0), (self.win_size[0]-98,450,self.standard_size[0],self.standard_size[1]))
        self.draw_text()

    def save_maze(self):
        obstacle_collection = []
        for obstacle in self.obstacle_list:
            obstacle_data = []
            obstacle_data.append(obstacle.width)
            obstacle_data.append(obstacle.height)
            obstacle_data.append(obstacle.rect.x)
            obstacle_data.append(obstacle.rect.y)
            obstacle_collection.append(obstacle_data)

        pickle_out = open("maze_files/maze3.pickle", "wb")
        pickle.dump(obstacle_collection, pickle_out)
        pickle_out.close()

    def event_handler(self, event = None):

        if event is not None:
            if event.type == pygame.MOUSEBUTTONDOWN:
                m_pos = pygame.mouse.get_pos()
                # Pick and Save rectangle
                if event.button == 1:
                    if self.rect.collidepoint(m_pos):
                        self.clicked_rect = True
                        self.angle = 0
                        self.size = self.standard_size.copy()
                    elif self.finish_creator.collidepoint(m_pos):
                        self.save_maze()
                        return False
                    else:
                        if self.clicked_rect:
                            self.clicked_rect = False
                            self.pos_obstacle(True)
                # Pickup placed rect
                elif event.button == 3:
                    for obstacle in self.obstacle_list:
                        if obstacle.rect.collidepoint(m_pos):
                            self.obstacle_list.remove(obstacle)
                            self.angle = obstacle.angle
                            self.clicked_rect = True
            # Rotate rectangle
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q and self.clicked_rect:
                    self.angle +=90
                elif event.key == pygame.K_e and self.clicked_rect:
                    self.angle -=90
            # Change Size
                elif event.key == pygame.K_w and self.clicked_rect:
                    self.size[0] += 15
                elif event.key == pygame.K_s and self.clicked_rect:
                    self.size[0] -= 15

    # Changed for 90 degree angle
    def pos_obstacle(self, set_pos = False):
        if self.clicked_rect:
            # Draw rect
            if (self.angle / 90) % 2 != 0:
                Obstacle([self.size[1], self.size[0]],self.angle).draw(self.screen)
            else:
                Obstacle(self.size, self.angle).draw(self.screen)

        if not self.clicked_rect and set_pos:
            # Save rect position
            if (self.angle / 90) % 2 != 0:
                Obstacle([self.size[1], self.size[0]], self.angle).add_to(self.obstacle_list)
            else:
                Obstacle(self.size, self.angle).add_to(self.obstacle_list)


    def draw_text(self):
        rotate_text = self.font.render('Click to pickup', True, (0, 0, 0))
        self.screen.blit(rotate_text, (self.win_size[0]-96,144))
        rotate_text = self.font.render('Obstacles placed: '+ str(self.obstacle_list.__len__()), True, (0, 0, 0))
        self.screen.blit(rotate_text, (self.win_size[0] - 125, 120))
        rotate_text = self.font.render('Press W and S to change size', True, (0, 0, 0))
        self.screen.blit(rotate_text, (self.win_size[0] - 175, 15))
        rotate_text = self.font.render('Press Q and E to rotate', True, (0, 0, 0))
        self.screen.blit(rotate_text, (self.win_size[0] - 140, 35))
        rotate_text = self.font.render('Save maze', True, (0, 0, 0))
        self.screen.blit(rotate_text, (self.win_size[0] - 90, 454))



class Obstacle(pygame.sprite.Sprite):
    def __init__(self, size, angle = 0):
        super().__init__()
        m_pos = pygame.mouse.get_pos()
        self.angle = angle
        self.image = pygame.Surface(size)
        self.image.fill((0, 255, 0))
        self.width = size[0]
        self.height = size[1]
        self.rect = self.image.get_rect()
        self.rect.x = m_pos[0] - (size[0] /2)
        self.rect.y = m_pos[1] - (size[1] /2)
        self.mask = None

    # Draw while positioning
    def draw(self, screen):
        screen.blit(self.image, self.rect.topleft)
    # Add to group to "save"
    def add_to(self,group):
        group.add(self)