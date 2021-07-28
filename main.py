from SelectScreen import *
from CreateObstacle import *
from TrainingEnvironment import TestEnvironment
pygame.init()

game_width = 640
game_height = 480
screen = pygame.display.set_mode([game_width, game_height])
clock = pygame.time.Clock()

# Incorporates all methods call needed to display all screens
class simulator_state:
    def __init__(self):
        self.state = "select_screen"
        self.run_simulator = True

        with open('bot_save/best_genome.pkl', 'rb') as f:
            pickle_data = pickle.load(f),
            self.best_genome = pickle_data[0][0]
            self.best_starts = pickle_data[0][1]
            self.best_fitnes = pickle_data[0][2]


    # Switches screens
    def draw_state(self):
        ### Code for select screen ###
        if self.state == "select_screen":
            # Draw here
            screen.fill((200, 200, 200))
            select_s.draw()
        ### Code or bot environment ###
        elif self.state == "train_envir":
            test_environment = TestEnvironment(20, 100, 500)
            test_environment.run_experiment()
        ### Code for obstacle creator ###
        elif self.state == "obstacle_creator":
            # Logic and drawing
            screen.fill((200, 200, 200))
            obstacle_creator.draw_selector()
            obstacle_creator.pos_obstacle()
            obstacle_creator.obstacle_list.draw(obstacle_creator.screen)

        elif self.state == "bot_replay":
            self.run_simulator = False
            t = TestEnvironment(1,1,500)
            t.generation_start_pos = self.best_starts
            t.evaluate_individual(self.best_genome,[[100,100],[100,100]])
            simulator.run_simulator = False
            pygame.quit()


    def handle_events(self,event):
        if self.state == "select_screen":
            if select_s.event_handler(event) is not None:
                simulator.state= select_s.event_handler(event)
        elif self.state == "obstacle_creator":
            # Event handler
            if obstacle_creator.event_handler(event) is not None:
                simulator.run_simulator = obstacle_creator.event_handler(event)




# Initialize variables / classes
simulator = simulator_state()
run_simulator = True
# Select screen
select_s = SelectScreen(screen)
# Obstacle creator
obstacle_creator = ObstacleCreator(screen)


########## Main Game Loop #########
while simulator.run_simulator:
    # Draw selected screen
    simulator.draw_state()

    if simulator.state == "bot_replay" or simulator.state == "train_envir":
        break

    # Framerate
    clock.tick(60)
    for event in pygame.event.get():
        # Close simulator
        if event.type == pygame.QUIT:
            simulator.run_simulator = False
        # Handle events for each screen
        simulator.handle_events(event)
    pygame.display.update()

pygame.quit()