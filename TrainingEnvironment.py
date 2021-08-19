import pygame
import numpy as np
import math
import pickle
from pygame.math import Vector2
import pygame.gfxdraw
import pandas as pd
from EvolutionaryAlgorithm import EvolutionaryAlgorithm
from NeuralNetwork import NeuralNetwork2


##########################################################################################################################################################################################


class differential_bot:
    def __init__(self, pos, radius, obstacle_list, sensor_length):
        # Obstacles
        self.obstacle_list = obstacle_list

        # Attributes
        self.pos = pos
        self.circle_radius = radius
        self.score = 0

        # Vector representation
        self.v_length = sensor_length + self.circle_radius
        self.max_v_length = self.v_length - self.circle_radius
        self.num_vectors = 12
        self.start_vector = Vector2(self.v_length,0)
        self.rotated_start_vector = self.start_vector
        self.vector_collection = []
        # Create initial distance vectors -> updated when angular speed != 0
        self.create_distance_vectors()
        self.mov_vect = np.array([0,0])

        # Differential drive
        self.theta = 0
        self.l_vel = 0
        self.r_vel = 0
        self.max_vel = 300
        self.min_vel = -50
        self.vel_increment = 5
        self.l = 20


    # Begin vector methods
    def create_distance_vectors(self):
        self.vector_collection.clear()
        angle_gap = 360 / self.num_vectors
        sum_angle = angle_gap
        self.vector_collection.append(self.rotated_start_vector)

        for _ in range(0,self.num_vectors -1):
            self.vector_collection.append(self.rotated_start_vector.rotate(sum_angle))
            sum_angle += angle_gap

    def get_distances(self):
        distances = []
        for vector in self.vector_collection:
            end_point = self.raycast_vector(vector)
            distances.append(round(end_point.distance_to(self.pos),3) - self.circle_radius)

        return distances

    def set_motor_speed_percentage(self, l_motor, r_motor):
        m_speed = self.max_vel
        self.r_vel = r_motor * m_speed
        self.l_vel = l_motor * m_speed

    def raycast_vector(self, vector):
        end_point = Vector2(self.pos)
        norm_vector = Vector2(vector).normalize()

        # Needed to check for point -> Mask Collision
        def translate_endpoint(u, v):
            return [int(u[i] - v[i]) for i in range(len(u))]

        for _ in range(self.v_length):
            end_point += norm_vector
            for obstacle in self.obstacle_list:
                # First check rect collision
                # If collision and obstacle has mask check mask collision
                if obstacle.rect.collidepoint(end_point):
                    if obstacle.mask is not None:
                        try:
                            rel_point = translate_endpoint(end_point, (obstacle.rect[0], obstacle.rect[1]))
                            if obstacle.mask.get_at(rel_point):
                                return end_point
                        except:
                            pass
                    else:
                        return end_point

        return end_point
    # End vector methods

    # Begin bot control methods
    def change_vel(self, event = None):
        # Neural Network Speed Logic


        #Max Speed restriction
        self.l_vel = min(self.l_vel, self.max_vel)
        self.r_vel = min(self.r_vel, self.max_vel)
        #Min speed restriction
        self.l_vel = max(self.l_vel, self.min_vel)
        self.r_vel = max(self.r_vel, self.min_vel)

    def calc_pos(self, seconds_elapsed):
        self.mov_vect = np.array([((self.l_vel + self.r_vel)/2) * math.cos(self.theta) * seconds_elapsed, -((self.l_vel + self.r_vel) / 2) * math.sin(self.theta) * seconds_elapsed])

        # Update position#
        self.theta += (self.r_vel - self.l_vel) / self.l * seconds_elapsed

        if self.theta > 2 * math.pi or self.theta < -2 * math.pi:
            self.theta = 0

        # Rotate
        self.rotated_start_vector = self.start_vector.rotate_rad(-self.theta)
        self.create_distance_vectors()

    def draw_vectors(self, screen):
        distances = []
        for vector in self.vector_collection:
            end_point = self.raycast_vector(vector)
            pygame.draw.line(screen, (128,0,128), self.pos, end_point ,2 )

    def update_score(self):
        v = (abs(self.l_vel) + abs(self.r_vel)) / 2 / self.max_vel
        d_v = 1 - abs(self.l_vel - self.r_vel) / (abs(self.min_vel) + self.max_vel)

        i = abs(min(self.get_distances())) / self.max_v_length
        score_this_frame = v * math.sqrt(d_v) * i
        self.score += score_this_frame
    # End bot control methods

    # Begin visual methods
    def draw_bot(self,screen):
        # Bot
        #pygame.draw.circle(self.screen, (0, 0, 255), (int(self.pos[0]), int(self.pos[1])), self.circle_radius)
        # Heading line
        x_axis = (self.pos[0] + self.circle_radius * math.cos(-self.theta), self.pos[1] + self.circle_radius * math.sin(-self.theta))
        pygame.draw.line(screen, (255, 0, 0), (self.pos[0], self.pos[1]), x_axis, 3)

##########################################################################################################################################################################################

class PlayerSprite(pygame.sprite.Sprite):
    def __init__(self, radius, bot):
        super().__init__()
        self.radius = radius / 2
        self.bot = bot
        self.image = pygame.Surface([radius * 2, radius * 2], pygame.SRCALPHA)
        self.rect = self.image.get_rect()
        self.rect.center = bot.pos
        pygame.gfxdraw.aacircle(self.image, radius, radius, radius, (0, 0, 255))
        pygame.gfxdraw.filled_circle(self.image, radius, radius, radius, (0, 0, 255))
        #self.image.set_colorkey((0, 0, 0))
        #self.mask = pygame.mask.from_surface(self.image)
        self.pGroup = pygame.sprite.Group()
        self.pGroup.add(self)

    def update(self):
        mov_vect = self.bot.mov_vect
        mov_vect_norm = mov_vect / np.linalg.norm(mov_vect)

        nextPos = [self.bot.pos[0] + mov_vect_norm[0], self.bot.pos[1] + mov_vect_norm[1]]
        self.rect.center = nextPos
        collisions = pygame.sprite.spritecollide(self, self.bot.obstacle_list, False)
        #collisions = self.check_collision()
        # Modify position when hitting a wall to not get stuck
        modify_x = 0
        modify_y = 0
        modify_by = 4

        if len(collisions) > 0:
            for col in collisions:
                if self.point_line_distance(self.bot.pos, col.rect.topleft,col.rect.topright) <= self.bot.circle_radius and self.bot.pos[1] <= col.rect.top:
                    #print("hitting top")
                    modify_y = -modify_by
                    obs_vect = np.array([col.width, 0])
                elif self.point_line_distance(self.bot.pos, col.rect.bottomleft,col.rect.bottomright) <= self.bot.circle_radius and self.bot.pos[1] >= col.rect.bottom:
                    #print("hitting bottom")
                    modify_y = modify_by
                    obs_vect = np.array([col.width, 0])
                elif self.point_line_distance(self.bot.pos, col.rect.bottomleft,col.rect.topleft) <= self.bot.circle_radius and self.bot.pos[0] <= col.rect.left:
                    #print("hitting left side")
                    modify_x = -modify_by
                    obs_vect = np.array([0, col.height])
                else:
                    #print("hitting right side")
                    modify_x = modify_by
                    obs_vect = np.array([0, col.height])
                    
                obs_vect = obs_vect / np.linalg.norm(obs_vect)
                mov_vect = obs_vect * np.dot(mov_vect, obs_vect) / np.linalg.norm(obs_vect)

                # Calculate the direction of the robot compared to the wall
                nr = abs((((col.rect.x + col.rect.width) - col.rect.x)*(self.bot.pos[1] - col.rect.y) - (self.bot.pos[0] - col.rect.x)*((col.rect.y + col.rect.height) - col.rect.y)))
                # 'normalize' it so wall length has no effect
                if col.rect.height > col.rect.width:
                    nr /= col.rect.height
                else:
                    nr /= col.rect.width

                if nr < 18:
                    mov_vect = [0,0]

        self.bot.pos[0] += mov_vect[0] +modify_x
        self.bot.pos[1] += mov_vect[1] +modify_y

        self.rect.center = self.bot.pos
        return len(collisions)

    def point_line_distance(self, pos, line_point1, line_point2):
        A = pos[0] - line_point1[0]
        B = pos[1] - line_point1[1]
        C = line_point2[0] - line_point1[0]
        D = line_point2[1] - line_point1[1]
        dot = A * C + B * D
        len_sq = C * C + D * D
        param = -1

        if len_sq != 0:
            param = dot / len_sq
        if param < 0:
            xx = line_point1[0]
            yy = line_point1[1]
        elif param > 1:
            xx = line_point2[0]
            yy = line_point2[1]
        else:
            xx = line_point1[0] + param * C
            yy = line_point1[1] + param * D

        dx = pos[0] - xx
        dy = pos[1] - yy
        return math.sqrt(dx * dx + dy * dy);

    def check_collision(self):
        collided_obstacles = []
        for obstacle in self.bot.obstacle_list:
            distances = []
            rect = obstacle.rect
            distances.append(self.point_line_distance(self.bot.pos, rect.topleft, rect.topright))
            distances.append(self.point_line_distance(self.bot.pos, rect.topleft, rect.bottomleft))
            distances.append(self.point_line_distance(self.bot.pos, rect.bottomleft, rect.bottomright))
            distances.append(self.point_line_distance(self.bot.pos, rect.bottomright, rect.topright))

            if min(distances) <= self.bot.circle_radius:
                collided_obstacles.append(obstacle)

        return collided_obstacles


##########################################################################################################################################################################################

class Obstacle(pygame.sprite.Sprite):
    def __init__(self,width, height, x, y, group):
        super().__init__()
        self.width = width
        self.height = height
        self.image = pygame.Surface([width, height])
        self.image.fill((0, 255, 0))
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y
        self.mask = None
        group.add(self)

##########################################################################################################################################################################################

class Dust(pygame.sprite.Sprite):
    def __init__(self, x, y, width, group):
        super().__init__()
        self.width = width
        self.height = width
        self.image = pygame.Surface([width, width])
        self.image.fill((0,50,150))
        self.rect = self.image.get_rect()
        self.rect.y = y
        self.rect.x = x
        group.add(self)


def setup_dust(spacing, game_width, game_height, obstacles, bot):
    dust_list = pygame.sprite.Group()
    for i in range(0, int(game_height / spacing)):
        for j in range(0, int(game_width / spacing)):
            y = i * spacing
            x = j * spacing
            width = spacing / 3
            generated_dust = Dust(x, y, width, dust_list)

            # Remove the dust if its generated in an obstacle
            if obstacles is not None:
                collisions = pygame.sprite.spritecollide(generated_dust, obstacles, False)
                if len(collisions) > 0:
                    dust_list.remove(generated_dust)

    # Remove all generated dust on player spawnpoint
    collisions = pygame.sprite.spritecollide(bot, dust_list, False)
    for dust in collisions:
        dust_list.remove(dust)

    return dust_list


def check_dust_robot_colision(dust_list, robot):
    collisions = pygame.sprite.spritecollide(robot, dust_list, False)

    pickup_amount = 0
    for dust in collisions:
        pickup_amount += 1
        dust_list.remove(dust)

    return pickup_amount

#########################################################################################################################################################################################




#########################################################################################################################################################################################

class TestEnvironment:
    def __init__(self, num_individuals, num_generations, testing_cycles):
        # Experiement parameters
        self.num_experiments_conducted = 0
        # Max distance detected by bot sensors
        self.sensor_length = 80
        self.delta_v = 0.05
        # Cycles after which the recurrent input gets updated
        self.delta_cycle_nn = 20

        # Genetic Algorithm Setup
        self.NN = NeuralNetwork2([6,4,2], 5, 12, self.delta_cycle_nn)
        self.mutation_rate = 0.3
        self.ev_alg = EvolutionaryAlgorithm(num_individuals , self.NN.create_genomes(num_individuals), self.mutation_rate)
        self.genomes = self.ev_alg.genomes

        self.num_individuals = num_individuals
        self.num_generations = num_generations
        self.testing_cycles = testing_cycles
        self.num_start_pos = 1

        # Test Environment
        self.win_size = [640,480]
        self.clock = pygame.time.Clock()

        # Test Subject / Bot & Test Maze
        self.obstacle_list = self.reconstruct_obstacles("maze_files/maze3.pickle")
        self.obstacle_list2 = self.reconstruct_obstacles("maze_files/maze2.pickle")
        self.test_mazes = [self.obstacle_list, self.obstacle_list2]

        self.pSprite = PlayerSprite(30, differential_bot([100,100], 30, self.obstacle_list, self.sensor_length))
        self.starting_pos = self.create_random_start()
        self.change_pos_every = 20

        # Stats & Data
        self.saved_evals = []
        self.saved_pos = []
        self.saved_genomes = []
        self.best_fitness = 0
        self.best_genome = []
        self.best_starts = []

        # visualize every xth experiement
        self.visualize_every = 100

        # Settings
        print("""####
        Number of generations: """+ str(self.num_generations)+"""
        Number of individuals: """+ str(self.num_individuals)+"""
        Number of cycles per experiment: """+ str(self.testing_cycles)+"""
        New starting positions every x generations: """+ str(self.change_pos_every)+"""
        Visualize every xth experiment: """+ str(self.visualize_every)+"""
        """)

    def reconstruct_obstacles(self, filename):
        pickle_in = open(filename, "rb")
        obstacle_data = pickle.load(pickle_in)
        obstacle_list = pygame.sprite.Group()
        for obstacle in obstacle_data:
            Obstacle(obstacle[0], obstacle[1], obstacle[2], obstacle[3], obstacle_list)

        return obstacle_list

    def create_random_start(self):
        starting_positions = []
        for obstacle_list in self.test_mazes:
            found_pos = False
            for _ in range(self.num_start_pos):
                while not found_pos:
                    # Introduce a padding so the bot does not spawn at the edge
                    border_with = 60
                    x = np.random.randint(border_with,self.win_size[0] -border_with)
                    y = np.random.randint(border_with,self.win_size[1] - border_with)

                    self.pSprite.bot.obstacle_list = obstacle_list
                    self.pSprite.bot.pos = [x,y]

                    # Determine if position is suitable by checking distance sensors
                    if min(self.pSprite.bot.get_distances()) > 10:
                        random_theta = np.random.uniform(0, 2 * math.pi)
                        starting_positions.append([x,y, random_theta])
                        found_pos = True
        return starting_positions

    def run_experiment(self):
        self.num_experiments_conducted = 0
        for generation_num in range(self.num_generations):
            evals = []
            print("####")
            print("Generation num: "+str(generation_num))
            print("####")
            # define new starting positions each generation
            if generation_num % self.change_pos_every == 0:
                self.generation_start_pos = self.create_random_start()

            #### Evaluation ####
            for i in range(self.num_individuals):
                print("individual num: "+str(i))

                genome = self.genomes[i]
                fittnes_value = self.evaluate_individual(genome, self.starting_pos)

                evals.append(fittnes_value)
                self.num_experiments_conducted += 1

                # Save data for each generation
            self.saved_evals.extend(evals)
            self.saved_pos.append(self.starting_pos)
            self.saved_genomes.extend(self.ev_alg.genomes)
            ####################

            # Apply evolutionary algorithm
            self.genomes = self.ev_alg.evolve(evals)


        # Save data to excel
        self.save_data()

    def evaluate_individual(self, genome, starting_positions):
        self.pSprite.bot.score = 0

        i = 0
        total_dust = 0
        num_collisions = 0
        for obstacle_list in self.test_mazes:

            # Give obstacles to bot
            self.pSprite.bot.obstacle_list = obstacle_list

            for test_num in range(len(starting_positions) // len(self.test_mazes)):
                # Let the bot spawn in different pos and average results i counts from zero to (len(test.mazes) * self.num_start_pos)-1
                self.pSprite.bot.pos = self.generation_start_pos[i][0:2]
                # set sprite center to bot pose so dust does not collide with old position -> would be chnaged on move
                self.pSprite.rect.center = self.generation_start_pos[i][0:2]
                self.pSprite.bot.theta = self.generation_start_pos[i][2]
                self.pSprite.bot.l_vel = 0
                self.pSprite.bot.r_vel = 0

                # Create new dust for each test case

                self.dust_list = setup_dust(10, self.win_size[0], self.win_size[1], obstacle_list, self.pSprite)
                starting_dust_am = len(self.dust_list)


                cycle_num = 0
                while cycle_num < self.testing_cycles:
                    # Calc NN
                    distance = self.pSprite.bot.get_distances()

                    motor_speed_perc = self.NN.calculate(distance, genome, cycle_num)
                    self.pSprite.bot.set_motor_speed_percentage((motor_speed_perc[0] * 2) - 1, (motor_speed_perc[1] * 2) - 1)

                    # Move bot
                    self.pSprite.bot.calc_pos(self.delta_v)
                    num_collisions+=self.pSprite.update()

                    # Check dust collected
                    check_dust_robot_colision(self.dust_list, self.pSprite)
                    #distance_fitness += max(self.NN.decay_distance(distance)) / 20
                    cycle_num += 1
                #catched_dust_perc += (1 - starting_dust_am / len(self.dust_list)) / self.testing_cycles
                i+=1
                total_dust+= starting_dust_am - len(self.dust_list)

        if self.num_experiments_conducted % self.visualize_every == 0:
            print("experiement_start_dust "+str(starting_dust_am))
            print("experiement_end_dust "+ str(len(self.dust_list)))
            self.visualize_test(genome)

        fitness_value = (total_dust - num_collisions * 10) / len(starting_positions)
        print("fitness "+str(fitness_value))
        print("###############")
        if fitness_value > self.best_fitness:
            self.best_fitness = fitness_value
            self.best_genome = genome
            self.best_starts = self.generation_start_pos
        return fitness_value

    def save_data(self):
        df = pd.DataFrame()
        df["generation"] = np.repeat(np.arange(1,self.num_generations+1), self.num_individuals)
        df["genome"] = self.saved_genomes
        df["evals"] = self.saved_evals
        df["starting_pos"] = [self.starting_pos] * self.num_individuals * self.num_generations
        df.to_excel("bot_save/data.xlsx")
        with open("bot_save/best_genome.pkl", 'wb') as f:
            pickle.dump([self.best_genome, self.best_starts, self.best_fitness], f)

    def visualize_test(self, genome):
        game_width = 640
        game_height = 480
        screen = pygame.display.set_mode([game_width, game_height])
        pygame.display.set_caption("test")
        clock = pygame.time.Clock()
        collected_dust = 0
        i = 0
        for obstacle_list in self.test_mazes:
            visualize_cycle = 0
            self.pSprite.bot.pos = self.generation_start_pos[i][0:2]
            self.pSprite.bot.theta = self.generation_start_pos[i][2]
            self.pSprite.rect.center = self.generation_start_pos[i][0:2]
            i+=1
            self.pSprite.bot.l_vel = 0
            self.pSprite.bot.r_vel = 0
            self.pSprite.bot.obstacle_list = obstacle_list
            self.dust_list = setup_dust(10, self.win_size[0], self.win_size[1], obstacle_list, self.pSprite)
            len_start_dust = len(self.dust_list)

            while visualize_cycle < self.testing_cycles:
                clock.tick(60)
                # Event -> close
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        visualize_cycle = self.testing_cycles
                # Viszalize
                screen.fill((200,200,200))
                obstacle_list.draw(screen)
                self.dust_list.draw(screen)
                # Automated Movement
                distance = self.pSprite.bot.get_distances()
                motor_speed_perc = self.NN.calculate(distance, genome, visualize_cycle)
                self.pSprite.bot.set_motor_speed_percentage((motor_speed_perc[0] * 2) - 1, (motor_speed_perc[1] * 2) - 1)
                self.pSprite.bot.calc_pos(self.delta_v)
                self.pSprite.update()

                self.pSprite.bot.draw_vectors(screen)
                self.pSprite.pGroup.draw(screen)
                self.pSprite.bot.draw_bot(screen)


                check_dust_robot_colision(self.dust_list, self.pSprite)

                pygame.display.update()
                visualize_cycle+=1

            collected_dust+= len_start_dust - len(self.dust_list)

        collected_dust = collected_dust / len(self.test_mazes)
        print("collected dust:  "+ str(collected_dust))
        pygame.display.quit()
#########################################################################################################################################################################################




