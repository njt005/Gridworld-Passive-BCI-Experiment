#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 10:12:02 2019

@author: nicktacca
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
from training_settings import yes_or_no

def experiment_observation(env, obs, hunter, prey, Q_optimal, 
                           exp_data, num_blocks, exp_num):
    
    """ This function has a single optimal agent observe a random/experiment
    policy agent. The observing agent gives a prediction error based on whether 
    the agent took the right or wrong step and how "wrong"/"right"they agree 
    with the action. This function returns prediction error per episode.
    """
    Q_delta_blocks = []
    prob_delta_blocks = []
    prob_delta_norm_blocks = []
    prob_error_blocks = []
    blocks = []
    episodes_blocks = []
    error_rate_blocks =[]
    dt_blocks = []
    
    for block_num in range(1, num_blocks+1):
        
        # Input error rate
        if block_num in (1, 4, 9):
            error_rate = 0
        elif block_num in (2, 6, 7):
            error_rate = 0.2
        else:
            error_rate = 0.4
    
        Q_delta_eps = []
        prob_delta_eps = []
        prob_delta_norm_eps = []
        prob_error_eps = []
        blocks_eps = []
        episodes_eps = []
        error_rate_eps =[]
        dt_eps = []
        
        #Unpack data
        if exp_data:
            data = np.loadtxt(
                    "experiment_data/visual/starting_coordinates/Experiment " \
                    + exp_num + "/hp_start_Block " + str(block_num) + ".dat")
            hunter_start_y = data[:, 0]
            hunter_start_x = data[:, 1]
            prey_start_y = data[:, 2]
            prey_start_x = data[:, 3]
            actions_filename = "experiment_data/visual/actions/Experiment " + \
            exp_num + "/actions_Block " + str(block_num) + ".dat"
            with open(actions_filename, "rb") as fp:
                actions = pickle.load(fp)
            num_episodes = len(hunter_start_y)
            dt_list = np.loadtxt(
                    "experiment_data/visual/time_steps/Experiment " + \
                    exp_num + "/dt_Block " + str(block_num) + ".dat")
        else:
            num_episodes = int(input(
                            "Input number of episodes agent will observe: "))
        
        for episode in range(1, num_episodes+1):
            
            dt = dt_list[episode-1]
            
            if exp_data:        
                # Initialize hunter starting location
                hunter.x = int(hunter_start_x[episode-1])
                hunter.y = int(hunter_start_y[episode-1])
                
                # Initialize prey starting location
                prey.x = int(prey_start_x[episode-1])
                prey.y = int(prey_start_y[episode-1])
                
                init_step_dist = env.step_dist(hunter, prey)
            else:
                while True:
                
                    # Reset new hunter starting location
                    hunter.x = np.random.choice(env.Ny)
                    hunter.y = np.random.choice(env.Nx)
                    
                    # Reset prey starting location
                    prey.x = np.random.choice(env.Ny)
                    prey.y = np.random.choice(env.Nx)
                    
                    # Make sure prey is at least 4 steps from hunter
                    init_step_dist = env.step_dist(hunter, prey)
                    if init_step_dist >= 4:
                        break
            
            # Generate an episode
            iter_episode = 0
            
            # Errors for a given episode
            Q_delta_episode = []
            prob_delta_episode = []
            prob_delta_norm_episode = []
            prob_error_episode = []
            blocks_episode = []
            episodes_episode = []
            error_rate_episode =[]
            dt_episode = []
            
            if exp_data:
                for action in actions[episode-1]:
                    sx, sy = env.determine_state(hunter, prey)
                    hunter.step(action)  # hunter moves
                    Q_delta, prob_delta, prob_delta_norm, prob_error = \
                    obs.determine_error(sx, sy, Q_optimal, action)
                    iter_episode += 1
                    Q_delta_episode.append(Q_delta)
                    prob_delta_episode.append(prob_delta)
                    prob_delta_norm_episode.append(prob_delta_norm)
                    prob_error_episode.append(prob_error)
                    blocks_episode.append(block_num)
                    episodes_episode.append(episode)
                    error_rate_episode.append(error_rate)
                    dt_episode.append(dt)
            else:
                while True:
                    sx, sy = env.determine_state(hunter, prey)
                    action = env.get_action(sx, sy, hunter) # get hunter action
                    hunter.step(action)  # hunter moves
                    Q_delta, prob_delta, prob_delta_norm = \
                    obs.determine_error(sx, sy, Q_optimal, action)
                    Q_delta_episode.append(Q_delta)
                    prob_delta_episode.append(prob_delta)
                    prob_delta_norm_episode.append(prob_delta_norm)
                    prob_error_episode.append(prob_error)
                    blocks_episode.append(block_num)
                    episodes_episode.append(episode)
                    error_rate_episode.append(error_rate)
                    dt_episode.append(dt)
            
            # Append errors for each episode
            Q_delta_eps.append(Q_delta_episode)
            prob_delta_eps.append(prob_delta_episode)
            prob_delta_norm_eps.append(prob_delta_norm_episode)
            prob_error_eps.append(prob_error_episode)
            blocks_eps.append(blocks_episode)
            episodes_eps.append(episodes_episode)
            error_rate_eps.append(error_rate_episode)
            dt_eps.append(dt_episode)
            
        # Make np.arrays
        Q_delta_eps = np.hstack(Q_delta_eps)
        prob_delta_eps = np.hstack(prob_delta_eps)
        prob_delta_norm_eps = np.hstack(prob_delta_norm_eps)
        prob_error_eps = np.hstack(prob_error_eps)
        blocks_eps = np.hstack(blocks_eps)
        episodes_eps = np.hstack(episodes_eps)
        error_rate_eps = np.hstack(error_rate_eps)
        dt_eps = np.hstack(dt_eps)
        
        # Append errors for each block of experiment
        Q_delta_blocks.append(Q_delta_eps)
        prob_delta_blocks.append(prob_delta_eps)
        prob_delta_norm_blocks.append(prob_delta_norm_eps)
        prob_error_blocks.append(prob_error_eps)
        blocks.append(blocks_eps)
        episodes_blocks.append(episodes_eps)
        error_rate_blocks.append(error_rate_eps)
        dt_blocks.append(dt_eps)
    
    # Make np.arrays
    Q_delta_blocks = np.hstack(Q_delta_blocks)
    prob_delta_blocks = np.hstack(prob_delta_blocks)
    prob_delta_norm_blocks = np.hstack(prob_delta_norm_blocks)
    prob_error_blocks = np.hstack(prob_error_blocks)
    blocks = np.hstack(blocks)
    episodes_blocks = np.hstack(episodes_blocks)
    error_rate_blocks = np.hstack(error_rate_blocks)
    dt_blocks = np.hstack(dt_blocks)
    
    return Q_delta_blocks, prob_delta_blocks, prob_delta_norm_blocks, \
            prob_error_blocks, blocks, episodes_blocks, error_rate_blocks, \
            dt_blocks

def multi_agents_observation(env, obs, hunter, prey):
    """ Multiple agents observe experiment/random actions. Plot of reward/
    policy vs steps taken is displayed.
    """  
    num_agents = int(input("Input number of observation agents (0 - 1000): "))
    Q_delta_agents = []
    prob_delta_agents = []
    prob_delta_norm_agents = []
    prob_error_agents = []
    blocks_agents = []
    episodes_agents = []
    error_rate_agents = []
    dt_agents = []
    
    exp_data = yes_or_no("Would you like to use experiment data?")
    if exp_data:
        num_blocks = 9
        exp_num = input("Input Experiment Number: ")
    else:
        num_blocks = 1
        exp_num = None
        
    save_data = yes_or_no("Would you like to save the observation data?")
    for agent in range(num_agents):
        # Loading optimal policy
        Q_optimal = np.loadtxt("training_data/Q_tables/Q" + str(agent+1) + \
                               ".dat")
        Q_optimal = Q_optimal.reshape((2*env.hunter_vision+1, 
                                   2*env.hunter_vision+1, 
                                   len(env.action_dict)))
        Q_delta_agent, prob_delta_agent, prob_delta_norm_agent, \
        prob_error_agent, blocks_agent, episodes_agent, error_rate_agent, \
        dt_agent = experiment_observation(env, obs, hunter, prey, Q_optimal, 
                                          exp_data, num_blocks, exp_num)
        
        # Append Q_delta and prob_delta lists
        Q_delta_agents.append(Q_delta_agent)
        prob_delta_agents.append(prob_delta_agent)
        prob_delta_norm_agents.append(prob_delta_agent)
        prob_error_agents.append(prob_error_agent)
        blocks_agents.append(blocks_agent)
        episodes_agents.append(episodes_agent)
        error_rate_agents.append(error_rate_agent)
        dt_agents.append(dt_agent)
    
    # Make matrix for each error item
    Q_delta_agents = np.vstack(Q_delta_agents)
    prob_delta_agents = np.vstack(prob_delta_agents)
    prob_delta_norm_agents = np.vstack(prob_delta_norm_agents)
    prob_error_agents = np.vstack(prob_error_agents)
    blocks_agents = np.vstack(blocks_agents)
    episodes_agents = np.vstack(episodes_agents)
    error_rate_agents = np.vstack(error_rate_agents)
    dt_agents = np.vstack(dt_agents)
    
    # Find average Q_delta and prob delta for all agents
    Q_delta_average = np.mean(Q_delta_agents, axis=0)
    prob_delta_average = np.mean(prob_delta_agents, axis=0)
    prob_delta_norm_average = np.mean(prob_delta_norm_agents, axis=0)
    prob_error_average = np.mean(prob_error_agents, axis=0)
    
    # Accessing image folder
    sns.set_palette("Set2")
    
    if save_data:
        if exp_data:
            image_folder = "observation_data/exp_data/Experiment " + \
                            exp_num + "/images/"
        else:
            image_folder = "observation_training_data/new_data/images/"
    
    errors_avg = [Q_delta_average, 
                  prob_delta_average, 
                  prob_delta_norm_average,
                  prob_error_average]
    
    errors = [Q_delta_agents, 
              prob_delta_agents, 
              prob_delta_norm_agents,
              prob_error_agents]
    
    error_name = ["\u0394Q", 
                  "\u0394Probabilities", 
                  "\u0394Probabilities_normalized",
                  "Predicted Error"]
    
    # Plotting
    for item, name in zip(errors_avg, error_name):
        # Plotting results
        plt.plot(item)
        plt.title("Average values over " + str(num_agents) + \
                  " observation agents")
        plt.xlabel("Trials")
        plt.ylabel(name)
        plt.grid()
        if save_data:
            plt.savefig(image_folder + name + ".png", format='png', dpi=500)
        plt.show()
    
    # Plotting all agents
    for item, name in zip(errors, error_name):
        plt.imshow(item, cmap='GnBu', aspect='auto', origin='lower')
        cbar = plt.colorbar()
        cbar.set_label(name, rotation=90)
        plt.title("Values over " + str(num_agents) + " observation agents")
        plt.xlabel("Trials")
        plt.ylabel("Number of Agents")
        plt.grid()
        if save_data:
            plt.savefig(image_folder + name + "_all.png", format='png', 
                        dpi=500)
        plt.show()
        
    # Plotting histograms
    num_bins = 100

    if save_data:
        folder_path = "observation_data/exp_data/Experiment " + exp_num
        Q_delta_agents_filename = folder_path + "/Q_delta_agents.dat"
        prob_delta_agents_filename = folder_path + "/prob_delta_agents.dat"
        prob_delta_norm_agents_filename = folder_path + \
                                        "/prob_delta_norm_agents.dat"
        prob_error_agents_filename = folder_path + "/prob_error_agents.dat"
        
        # Average data as well
        average_data = [blocks_agents[0,:],
                  episodes_agents[0,:],
                  error_rate_agents[0,:],
                  dt_agents[0,:], 
                  errors_avg]
        
        errors_average = np.vstack(average_data)
        
        zero_error = []
        twenty_error = []
        forty_error = []
    
        for i in range(len(errors_average[7,:])):
            if errors_average[2,i] == 0:
                zero_error.append(errors_average[7,i])
            elif errors_average[2,i] == 0.2:
                twenty_error.append(errors_average[7,i])
            elif errors_average[2,i] == 0.4:
                forty_error.append(errors_average[7,i])
        
        prob_error_sorted = [zero_error, twenty_error, forty_error]
        
        error_labels= ["0%","20%", "40%"]
        n2, bins2, patches2 = plt.hist(prob_error_sorted, num_bins, 
                                       label=error_labels)
        plt.title("Histogram of Average Agent Prediction Errors")
        plt.xlabel("Prediction Error")
        plt.ylabel("Quantity")
        plt.grid()
        plt.legend()
        plt.show()
        
        errors_average = np.transpose(errors_average)
        
        errors_average_filename = folder_path + "/errors_average.dat"
        prob_error_average_filename = folder_path + "/prob_error_average.dat"
        prob_delta_average_filename = folder_path + "/prob_delta_average.dat"
        
        np.savetxt(Q_delta_agents_filename, 
                   Q_delta_agents, 
                   fmt='%-7.3f', 
                   header = "Delta Q for all Agents")
        
        np.savetxt(prob_delta_agents_filename, 
                   prob_delta_agents, 
                   fmt='%-7.3f', 
                   header = "Delta Probability for all Agents")
        
        np.savetxt(prob_delta_norm_agents_filename, 
                   prob_delta_norm_agents, 
                   fmt='%-7.3f', 
                   header = "Delta Probability Normalized for all Agents")
        
        # Prediction error
        
        prob_error_agents = [blocks_agents[0,:],
                             episodes_agents[0,:],
                             error_rate_agents[0,:],
                             dt_agents[0,:], 
                             prob_error_agents]
        prob_error_agents = np.vstack(prob_error_agents)
        
        prob_error_agents = np.transpose(prob_error_agents)
        
        np.savetxt(prob_error_agents_filename,
                   prob_error_agents,
                   fmt='%-7.3f', 
                   header = "Predicted Error")
        
        np.savetxt(errors_average_filename,
                   errors_average,
                   fmt='%-7.3f', 
                   header = "Average Errors")
        
        np.savetxt(prob_error_average_filename,
                   prob_error_average,
                   fmt='%-7.3f')
        
        np.savetxt(prob_delta_average_filename,
                   prob_delta_average,
                   fmt='%-7.3f')