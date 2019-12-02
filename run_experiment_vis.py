#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 10:56:29 2019

@author: nicktacca
"""
#%%
from gridworld_settings import env, hunter, prey
from experiment_settings import MapSim, ec, experiment_block, quit_experiment
from training_settings import yes_or_no

if __name__ == "__main__":
    
    new_exp = yes_or_no("Would you like to run a new experiment?")
    
    if new_exp:
        data_folder = None
        num_episodes = int(input("Input number of episodes per block: "))
    else:
        data_folder = input("Input experiment number: ")
        num_episodes = None
    
    save_data = yes_or_no("Would you like to save the experiment data?")
    
    # Initiate pygame display/map
    map_sim = MapSim(ec)
  
    # Define hunter position on grid
    map_sim.grid[hunter.y][hunter.x].append(hunter)
        
    # Define prey position on grid
    if (hunter.x, hunter.y) != (prey.x, prey.y):
        map_sim.grid[prey.y][prey.x].append(prey)
    
    # Run Experiments
    experiment_block(env, map_sim, hunter, prey, num_episodes, save_data,
                     new_exp, data_folder, ec.BLOCK1, 0, 1200)
    experiment_block(env, map_sim, hunter, prey, num_episodes, save_data, 
                     new_exp, data_folder, ec.BLOCK2, 20, 1200)
    experiment_block(env, map_sim, hunter, prey, num_episodes, save_data,
                     new_exp, data_folder, ec.BLOCK3, 40, 1200)
    experiment_block(env, map_sim, hunter, prey, num_episodes, save_data, 
                     new_exp, data_folder, ec.BLOCK4, 0, 800)
    experiment_block(env, map_sim, hunter, prey, num_episodes, save_data, 
                     new_exp, data_folder, ec.BLOCK5, 40, 800)
    experiment_block(env, map_sim, hunter, prey, num_episodes, save_data, 
                     new_exp, data_folder, ec.BLOCK6, 20, 800)
    experiment_block(env, map_sim, hunter, prey, num_episodes, save_data, 
                     new_exp, data_folder, ec.BLOCK7, 20, fixed_dt=False)
    experiment_block(env, map_sim, hunter, prey, num_episodes, save_data, 
                     new_exp, data_folder, ec.BLOCK8, 40, fixed_dt=False)
    experiment_block(env, map_sim, hunter, prey, num_episodes, save_data, 
                     new_exp, data_folder, ec.BLOCK9, 0, fixed_dt=False)
    quit_experiment()