import pygame

class SelectScreen:
    def __init__(self, screen):
        self.display_hint = False
        self.screen = screen
        smallfont = pygame.font.SysFont('Corbel', 20)
        pygame.display.set_caption("Differential Bot Environment")
        self.train_text= smallfont.render('Train Bot', True, (0, 0, 0))
        self.test_envir_text_rect = self.train_text.get_rect(center=(225, 240))
        self.obstacle_text= smallfont.render('Obstacle Creator', True, (0,0,0))
        self.obstacle_text_rect = self.obstacle_text.get_rect(center=(415, 240))
        self.maze_hint_text= smallfont.render('You have to create a maze first', True, (0,0,0))
        self.maze_hint_text_rect = self.train_text.get_rect(center=(280, 380))

        self.replay_envir_text = smallfont.render('Replay Genome', True, (0,0,0))
        self.replay_envir_text_rect = self.train_text.get_rect(center=(315, 320))
        # Needs to be in init as it also creates rect object
        self.go_train = pygame.draw.rect(screen, (255, 0, 0), [150, 220, 150, 40])
        self.go_obstacle_creator = pygame.draw.rect(screen, (0, 255, 0), [340, 220, 150, 40])
        self.go_replay = pygame.draw.rect(screen, (0, 0, 255), [340, 300, 150, 40])

    def event_handler(self,event):
        if event is not None:
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                m_pos = pygame.mouse.get_pos()
                if self.go_train.collidepoint(m_pos):
                    return "train_envir"
                elif self.go_obstacle_creator.collidepoint(m_pos):
                    return "obstacle_creator"
                elif self.go_replay.collidepoint(m_pos):
                        return "bot_replay"




    def draw(self):
        pygame.draw.rect(self.screen, (255, 0, 0), [150, 220, 150, 40])
        pygame.draw.rect(self.screen, (0, 255, 0), [340, 220, 150, 40])
        pygame.draw.rect(self.screen, (0, 150, 40), [265, 300, 150, 40])

        self.screen.blit(self.train_text, self.test_envir_text_rect)
        self.screen.blit(self.obstacle_text, self.obstacle_text_rect)
        self.screen.blit(self.replay_envir_text, self.replay_envir_text_rect)

        if self.display_hint:
            self.screen.blit(self.maze_hint_text, self.maze_hint_text_rect)