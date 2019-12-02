#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 10:27:00 2019

@author: Nick Tacca
"""
# %%
import numpy as np
import pygame as pygame
import pickle
import sys
from gridworld_settings import Agent, env, hunter, prey

class MapSim:

    def __init__(self, ec):
        pygame.init()

        # Define object names in simulation
        self.blank_name = ec.BLANK_NAME

        # Define grid definitions
        self.tile_width = 150
        self.tile_height = 150
        self.tile_margin = 10
        self.tile_size = self.tile_width + self.tile_margin
        self.display_height = 810
        self.display_width = 810

        # Font definitions
        self.font_small = pygame.font.SysFont("Arial", 50)
        self.font_med = pygame.font.SysFont("Arial", 100)
        self.font_large = pygame.font.SysFont("Arial", 150)
        self.font_xlarge = pygame.font.SysFont("Arial", 300)

        # Color definitions
        self.black = 0, 0, 0
        self.white = 255, 255, 255
        self.green = 0, 255, 0
        self.red = 255, 0, 0
        self.blue = 0, 0, 255

        # Pygame visualization settings
        self.clock = pygame.time.Clock()
        self.screen = pygame.display.set_mode(
            [self.display_height, self.display_width])

        # Initialize grid
        self.grid = [[[] for column in range(env.Ny)] for row in range(env.Nx)]

        # Completely fill grid with blank tiles
        for row in range(env.Nx):
            for column in range(env.Ny):
                self.grid[row][column].append(
                    Agent(self.blank_name, column, row, env))

        del row, column

    def text_objects(self, text, color, size):
        if size == "small":
            text_surface = self.font_small.render(text, True, color)
        elif size == "medium":
            text_surface = self.font_medium.render(text, True, color)
        elif size == "large":
            text_surface = self.font_large.render(text, True, color)
        return text_surface, text_surface.get_rect()

    def message_to_screen(self, message, color, y_displace=0, size="medium"):
        text_surface, text_rect = self.text_objects(message, color, size)
        text_rect.center = (self.display_width /
                            2), (self.display_height/2 + y_displace)
        self.screen.blit(text_surface, text_rect)

    def update(self):
        """Check every object in the grid and see if it has moved and 
        put it in its proper place by deleting it from its old list 
        and appending it to the proper location.
        """
        for row in range(env.Nx):
            for column in range(env.Ny):
                for i, object in enumerate(self.grid[row][column]):
                    if object.x != column or object.y != row:
                        self.grid[row][column].pop(i)
                        self.grid[object.y][object.x].append(object)

    def draw_grid(self):
        # Draw grid
        for row in range(env.Nx):
            for column in range(env.Ny):
                # Determine color based on objects in this spot
#                names = {object.name for object in self.grid[row][column]}
                color = self.white  # default color for blank tiles
#                if hunter.name in names:
#                    color = self.blue
#                elif prey.name in names:
#                    color = self.green

                pygame.draw.rect(self.screen, color,
                                 [self.tile_size*column + self.tile_margin,
                                  self.tile_size*row + self.tile_margin,
                                  self.tile_width,
                                  self.tile_height])
                if (hunter.x, hunter.y) != (prey.x, prey.y):
                    # Prey marker
                    self.screen.blit(self.font_large.render("X", True, 
                                                            self.red),
                                     (prey.x*self.tile_size + \
                                      3.5*self.tile_margin,
                                      prey.y*self.tile_size + \
                                      0.4*self.tile_margin))
                    
                    # Hunter marker
                self.screen.blit(self.font_xlarge.render(u"\u25CF", True, 
                                                         self.blue),
                                 (hunter.x*self.tile_size - \
                                  0.6*self.tile_margin,
                                  hunter.y*self.tile_size - \
                                  10.1*self.tile_margin))

class ExperimentConstants:
    
    # Blank Tile Name
    BLANK_NAME = "Blank"
    
    # Experiment Block Names
    BLOCK1 = "Block 1"
    BLOCK2 = "Block 2"
    BLOCK3 = "Block 3"
    BLOCK4 = "Block 4"
    BLOCK5 = "Block 5"
    BLOCK6 = "Block 6"
    BLOCK7 = "Block 7"
    BLOCK8 = "Block 8"
    BLOCK9 = "Block 9"

def pause():
    paused = True
    while paused:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit_experiment()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = False
                elif event.key == pygame.K_ESCAPE:
                    quit_experiment()


def automatic_pause(map_sim, dt):
    pygame.event.clear()
    blank_screen = pygame.USEREVENT + 1
    auto_start = pygame.USEREVENT + 2
    pygame.time.set_timer(blank_screen, dt)
    pygame.time.set_timer(auto_start, 4*dt)
    paused = True
    while paused:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit_experiment()
            elif event.type == blank_screen:
                map_sim.screen.fill(map_sim.black)
                pygame.display.flip()
            elif event.type == auto_start:
                paused = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    pause()
                elif event.key == pygame.K_ESCAPE:
                    quit_experiment()


def quit_experiment():
    pygame.quit()
    sys.exit()

def experiment_block(env, map_sim, hunter, prey, num_episodes, save_data, 
                     new_exp, data_folder, name, error, dt=None, 
                     fixed_dt=True, visual=True):
    """ 
    This function runs a single block in the experiment.
    """
    # Loading optimal policy
    Q_optimal = np.loadtxt("training_data/Q_Optimal.dat")
    Q_optimal = Q_optimal.reshape((2*env.hunter_vision+1,
                                   2*env.hunter_vision+1,
                                   len(env.action_dict)))

    # Movement directions
    movement_keys = {pygame.K_LEFT, pygame.K_RIGHT, pygame.K_UP, pygame.K_DOWN}
    
    # Total step counter
    num_steps = 0
    additional_steps = 0
    wrong_steps_total = 0
    wrong_episode = 0

    # Start with pause window
    map_sim.screen.fill(map_sim.white)
    map_sim.message_to_screen(name,
                              map_sim.black,
                              y_displace=-120,
                              size="large")
    map_sim.message_to_screen("Press space-bar to begin " + name,
                              map_sim.black,
                              y_displace=10,
                              size="small")
    map_sim.clock.tick(60)
    pygame.display.flip()
    pause()
    if visual:
        automatic_pause(map_sim, dt=1000)

    # New block
    print(name)

    if new_exp:
        # Starting locations for episodes
        hunter_start_coords = []
        prey_start_coords = []
    
        # Actions for all episodes
        actions_episodes = []
    
        # Time-steps per episode
        dt_episodes = []

        for eps in range(num_episodes):
    
            # Automatic move settings
            automatic_move = pygame.USEREVENT
    
            if not fixed_dt:
                dt_list = np.arange(800, 1300, 100)
                random_dt = np.random.choice(dt_list)
                pygame.time.set_timer(automatic_move, random_dt)
                dt_episodes.append(random_dt)
            else:
                pygame.time.set_timer(automatic_move, dt)
                dt_episodes.append(dt)
    
            # Generate a new episode
            iter_episode = 0
            wrong_steps = 0
    
            # Initialize simulation in episode
            pygame.init()
    
            while True:
    
                # Reset hunter starting location
                hunter.x = np.random.choice(env.Ny)
                hunter.y = np.random.choice(env.Nx)
                hunter_start = [hunter.y, hunter.x]
                map_sim.update()
    
                # Reset prey starting location
                prey.x = np.random.choice(env.Ny)
                prey.y = np.random.choice(env.Nx)
                prey_start = [prey.y, prey.x]
                map_sim.update()
    
                # Make sure prey is at least 4 steps from hunter
                init_step_dist = env.step_dist(hunter, prey)
                if init_step_dist >= 4:
                    break
                
            if visual:
                # Begin visualization
                map_sim.draw_grid()
                pygame.display.flip()
        
                # Calculate maximum possible reward for episode
                max_reward = env.total_reward(init_step_dist)
        
                # Actions in single episode
                actions_ep = []
        
                while True:
        
                    # Look for events in queue
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            quit_experiment()
        
                        # Automatic movement
                        elif event.type == automatic_move:
        
                            # Get action and states from optimal policy
                            sx_exp, sy_exp = env.determine_state(hunter, prey)
                            action_exp = env.get_experiment_action(sx_exp,
                                                                   sy_exp,
                                                                   Q_optimal,
                                                                   error,
                                                                   hunter)
                            dist_old_exp = env.step_dist(hunter, prey)
                            hunter.step(action_exp)  # Hunter moves
                            # determine new prey distance
                            dist_new_exp = env.step_dist(hunter, prey)
        
                            if dist_new_exp > dist_old_exp:
                                wrong_steps += 1
        
                            map_sim.update()
                            iter_episode += 1
        
                            # Store actions for episode
                            actions_ep.append(action_exp)
        
                        # Manual movement
                        elif event.type == pygame.KEYDOWN:
                            if event.key in movement_keys:
                                if event.key == pygame.K_UP:
                                    hunter.step(0)
                                elif event.key == pygame.K_RIGHT:
                                    hunter.step(1)
                                elif event.key == pygame.K_DOWN:
                                    hunter.step(2)
                                elif event.key == pygame.K_LEFT:
                                    hunter.step(3)
                                map_sim.update()
                            elif event.key == pygame.K_ESCAPE:
                                quit_experiment()
                            elif event.key == pygame.K_SPACE:
                                pause()
        
                    # Visualization updates
                    map_sim.screen.fill(map_sim.black)
                    map_sim.draw_grid()
                    map_sim.clock.tick(60)
                    pygame.display.flip()
        
                    # Reaching the target
                    # determine new prey distance
                    dist_new_exp = env.step_dist(hunter, prey)
                    if dist_new_exp == 0:
                        if visual:
                            automatic_pause(map_sim, dt=1000)
                        break
            else:    
                # Calculate maximum possible reward for episode
                max_reward = env.total_reward(init_step_dist)
        
                # Actions in single episode
                actions_ep = []
        
                # Assigning initial step distance
                dist_old_exp = init_step_dist
        
                while True:
                    # Get action and states from optimal policy
                    sx_exp, sy_exp = env.determine_state(hunter, prey)
                    action_exp = env.get_experiment_action(sx_exp,
                                                           sy_exp,
                                                           Q_optimal,
                                                           error,
                                                           hunter)
                    hunter.step(action_exp)  # Hunter moves
                    dist_new_exp = env.step_dist(hunter, prey)
                    iter_episode += 1
        
                    if dist_new_exp > dist_old_exp:
                        wrong_steps += 1
        
                    # Store actions for episode
                    actions_ep.append(action_exp)
        
                    # Terminate when hunter reaches prey
                    if dist_new_exp == 0:
                        break
        
                    # Update step distance to prey
                    dist_new_exp = dist_old_exp
                    
            # Calculate reward loss/wrong steps per episode
            reward_episode = env.total_reward(iter_episode)
            iter_loss = max_reward - reward_episode
    
            # Total number of steps/additional steps/wrong steps
            num_steps += iter_episode
            wrong_steps_total += wrong_steps
            additional_steps += iter_loss
            if iter_loss > 0:
                wrong_episode += 1
    
            # Print number of wrong steps per episode
            print("Number of wrong steps in episode {}: {}".format(eps + 1, 
                  wrong_steps))
    
            # Store starting hunter and prey coordinates and actions
            hunter_start_coords.append(hunter_start)
            prey_start_coords.append(prey_start)
            actions_episodes.append(actions_ep)
    else:
        if visual:
            data = np.loadtxt(
            "experiment_data/visual/starting_coordinates/Experiment " + \
            data_folder + "/hp_start_" + name + ".dat")
            actions_filename = "experiment_data/visual/actions/Experiment " + \
            data_folder + "/actions_" + name + ".dat"
            dt_list = np.loadtxt(
                    "experiment_data/visual/time_steps/Experiment " + \
                    data_folder + "/dt_" + name + ".dat")
        else:
            data = np.loadtxt(
            "experiment_data/no_visual/starting_coordinates/Experiment " + \
            data_folder + "/hp_start_" + name + ".dat")
            actions_filename = "experiment_data/no_visual/actions/Experiment "\
            + data_folder + "/actions_" + name + ".dat"
            dt_list = np.loadtxt(
                    "experiment_data/visual/time_steps/Experiment " + \
                    data_folder + "/dt_" + name + ".dat")
        hunter_start_y = data[:, 0]
        hunter_start_x = data[:, 1]
        prey_start_y = data[:, 2]
        prey_start_x = data[:, 3]
        with open(actions_filename, "rb") as fp:
            actions = pickle.load(fp)
        num_episodes = len(hunter_start_y)
        
        for episode in range(num_episodes):
            
            # Automatic move settings
            automatic_move = pygame.USEREVENT
            pygame.time.set_timer(automatic_move, int(dt_list[episode]))
           
            # Initialize hunter starting location
            hunter.x = int(hunter_start_x[episode])
            hunter.y = int(hunter_start_y[episode])
            
            # Initialize prey starting location
            prey.x = int(prey_start_x[episode])
            prey.y = int(prey_start_y[episode])
            
            init_step_dist = env.step_dist(hunter, prey)
            
            # Generate an episode
            iter_episode, old_iter_episode, wrong_steps = 0, 0, 0
            
            if visual:
                # Begin visualization
                map_sim.draw_grid()
                pygame.display.flip()
            
                # Calculate maximum possible reward for episode
                max_reward = env.total_reward(init_step_dist)
                
                # Assigning initial step distance
                dist_old = init_step_dist
                    
                for action in actions[episode]:
                    
                    old_iter_episode = iter_episode

                    while True:
                        
                        # Look for events in queue
                        for event in pygame.event.get():
                            if event.type == pygame.QUIT:
                                quit_experiment()
            
                            # Automatic movement
                            elif event.type == automatic_move:
                                hunter.step(action)  # hunter moves
                                dist_new = env.step_dist(hunter, prey)
                                iter_episode += 1
                                if dist_new > dist_old:
                                    wrong_steps += 1
                                # Update step distance to prey
                                dist_old = dist_new
                                break
        
                            # Manual movement
                            elif event.type == pygame.KEYDOWN:
                                if event.key in movement_keys:
                                    if event.key == pygame.K_UP:
                                        hunter.step(0)
                                    elif event.key == pygame.K_RIGHT:
                                        hunter.step(1)
                                    elif event.key == pygame.K_DOWN:
                                        hunter.step(2)
                                    elif event.key == pygame.K_LEFT:
                                        hunter.step(3)
                                    map_sim.update()
                                elif event.key == pygame.K_ESCAPE:
                                    quit_experiment()
                                elif event.key == pygame.K_SPACE:
                                    pause()
            
                        # Visualization updates
                        map_sim.screen.fill(map_sim.black)
                        map_sim.draw_grid()
                        map_sim.clock.tick(60)
                        pygame.display.flip()
            
                        # Reaching the target
                        # determine new prey distance
                        if iter_episode > old_iter_episode:
                            dist_new_exp = env.step_dist(hunter, prey)
                            if dist_new_exp == 0:
                                if visual:
                                    automatic_pause(map_sim, dt=1000)
                                    break
                            else:
                                break
                        
                        old_iter_episode = iter_episode
            else:    
                # Calculate maximum possible reward for episode
                max_reward = env.total_reward(init_step_dist)
        
                # Assigning initial step distance
                dist_old_exp = init_step_dist
                
                for action in actions[episode-1]:
                    hunter.step(action)  # hunter moves
                    dist_new = env.step_dist(hunter, prey)
                    iter_episode += 1
                    if dist_new > dist_old:
                        wrong_steps += 1
                    # Terminate if hunter reaches the prey
                    if dist_new == 0:
                        break
                    # Update step distance to prey
                    dist_old = dist_new
                    
            # Calculate reward loss/wrong steps per episode
            reward_episode = env.total_reward(iter_episode)
            iter_loss = max_reward - reward_episode
    
            # Total number of steps/additional steps/wrong steps
            num_steps += iter_episode
            wrong_steps_total += wrong_steps
            additional_steps += iter_loss
            if iter_loss > 0:
                wrong_episode += 1

            # Print number of wrong steps per episode
            print("Number of wrong steps in episode {}: {}".format(episode + 1,
                  wrong_steps))
            
    # Calculate percent wrong of agent
    percent_wrong_steps = wrong_steps_total/num_steps
    percent_additional_steps = additional_steps/num_steps
    percent_wrong_episodes = wrong_episode/num_episodes
    print("\nInput Error Percent: {:.2F}%".format(error))
    print("Step Error Percent: {:.2F}%".format(percent_wrong_steps*100))
    print("Additional Step Percent: {:.2F}%".format(
        percent_additional_steps*100))
    print("Episode Error Percent: {:.2F}%\n".format(
        percent_wrong_episodes*100))

    # Starting coordinates per episode
    if save_data:
        start_stacked = np.column_stack((hunter_start_coords, 
                                         prey_start_coords))
        header_start = "Hunter Starting Coordinates, Prey Starting Coordinates"
        if visual:
            start_filename = \
            "experiment_data/visual/starting_coordinates/hp_start_" + \
            name + ".dat"
            np.savetxt(start_filename, start_stacked, fmt='%s', 
                       header=header_start)
        else:
            start_filename = \
            "experiment_data/no_visual/starting_coordinates/hp_start_" + \
            name + ".dat"
            np.savetxt(start_filename, start_stacked, fmt='%s', 
                       header=header_start)

    # Actions per episode
    if save_data:
        header_actions = "Actions / Episode"
        if visual:
            actions_filename = "experiment_data/visual/actions/actions_" + \
            name + ".dat"
            with open(actions_filename, "wb") as fp:
                pickle.dump(actions_episodes, fp)
        else:
            actions_filename = "experiment_data/no_visual/actions/actions_" + \
            name + ".dat"
            with open(actions_filename, "wb") as fp:
                pickle.dump(actions_episodes, fp)

    # Time-step per episode
    if save_data:
        dt_episodes = np.array(dt_episodes)
        header_actions = "Time-step / Episode"
        if visual:
            dt_filename = "experiment_data/visual/time_steps/dt_" + \
            name + ".dat"
            np.savetxt(dt_filename, 
                       dt_episodes, 
                       fmt='%s', 
                       header=header_actions)
        else:
            dt_filename = "experiment_data/no_visual/time_steps/dt_" + \
            name + ".dat"
            np.savetxt(dt_filename, 
                       dt_episodes, 
                       fmt='%s', 
                       header=header_actions)

# Only initiate constants to prevent Pygame window from popping up early
ec = ExperimentConstants()

