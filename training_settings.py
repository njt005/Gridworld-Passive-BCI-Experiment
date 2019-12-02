#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 10:12:02 2019

@author: nicktacca
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def optimal_policy(env, hunter, prey, num_episodes, show_plots):
    
    """ This function trains a single optimal policy agent.
    A Q table with percent error & number of episodes is returned 
    for each agent. Convergence criteria is defined as when the 
    rolling mean of max policy change (Window size = 100) < 1e-6.
    """
    steps_episode = []
    loss_episode = []
    wrong_steps_episode = []
    percent_wrong_steps_episode = []
    policy_delta = []
    
    # Initialize Q table per episode
    env.Q = np.zeros((2 * env.hunter_vision + 1, 
                                 2 * env.hunter_vision + 1, 
                                 len(env.action_dict)))
    
    # Initialize old Q table for all episodes
    Q_old = np.zeros((num_episodes, 
                      2 * env.hunter_vision + 1, 
                      2 * env.hunter_vision + 1,
                      len(env.action_dict)))
    
    for episode in range(1, num_episodes+1):
        
        while True:
        
            # Reset hunter starting location
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
        iter_episode, reward_episode, wrong_steps = 0, 0, 0
        
        # Calculate maximum possible reward for episode
        max_reward = env.total_reward(init_step_dist)
        
        # Assigning initial step distance
        dist_old = init_step_dist
        
        # Update policy
        Q_old[episode-1,:,:,:] = env.Q
        
        while True:
            sx, sy = env.determine_state(hunter, prey) # determine hunter state
            action = env.get_action(sx, sy, hunter)  # get hunter action
            hunter.step(action)  # hunter moves
            sx_, sy_ = env.determine_state(hunter, prey) 
            dist_new = env.step_dist(hunter, prey)
            reward = env.give_reward(dist_new) # give reward to hunter
            env.train(sx, sy, action, reward, sx_, sy_)  # train agent
            iter_episode += 1
            if dist_new > dist_old:
                wrong_steps += 1
            reward_episode += reward
            # Terminate if hunter reaches the prey
            if dist_new == 0:
                break
            if wrong_steps > 1000:
                break
            # Update step distance to prey
            dist_old = dist_new
            
        # Determine loss for episode    
        iter_loss = max_reward - reward_episode
            
        # Steps/Loss/Wrong step count
        WINDOW_SIZE_ERROR_RATE = 1000
        steps_episode.append(iter_episode)
        loss_episode.append(iter_loss)
        wrong_steps_episode.append(wrong_steps)
        
        # Percent wrong steps
        percent_wrong_steps = wrong_steps/iter_episode
        percent_wrong_steps_episode.append(percent_wrong_steps)
        percent_wrong_steps_mean = rolling_mean(percent_wrong_steps_episode, 
                                                WINDOW_SIZE_ERROR_RATE)
        
        # Policy congergence
        WINDOW_SIZE_POLICY = 100
        iter_policy_delta = np.max(np.abs(np.subtract(Q_old[episode-1], 
                                                      env.Q)))
        policy_delta.append(iter_policy_delta)
        pd_rolling_mean = rolling_mean(policy_delta, WINDOW_SIZE_POLICY)
        
        # Print episodes
        if (episode == 0) or (episode + 1) % WINDOW_SIZE_ERROR_RATE == 0 \
        and len(pd_rolling_mean) == 0:
            print("[Ep {}] -> Number of Wrong Steps = {:.1F} -> Rolling Mean Policy Change = N/A ".format(
                episode + 1, wrong_steps))
        elif (episode == 0) or (episode + 1) % WINDOW_SIZE_ERROR_RATE == 0 \
        and len(pd_rolling_mean) > 0:
            print("[Ep {}] -> Number of Wrong Steps = {:.1F} -> Rolling Mean Policy Change = {:.6F} ".format(
                episode + 1, iter_loss, pd_rolling_mean[episode-WINDOW_SIZE_POLICY]))
        
        if episode > 1000:
            if pd_rolling_mean[episode-WINDOW_SIZE_POLICY] < 1e-6:
                total_train_episodes = episode + 1
                print("Hunter is fully trained!")
                print("\nTotal episodes: {}".format(total_train_episodes))
                hunter_error_rate = \
                percent_wrong_steps_mean[episode-WINDOW_SIZE_ERROR_RATE]
                print("Hunter error rate: {:.2F}%".format(
                        hunter_error_rate*100))
                hunter_train_data = np.array([total_train_episodes, 
                                              hunter_error_rate])
                break
        
    if show_plots:
        # Plotting steps & loss
        line1, = plt.plot(steps_episode, label = "Steps")
        line2, = plt.plot(loss_episode, label = "Reward Loss")
        line3, = plt.plot(wrong_steps_episode, label = "Wrong Steps")
        legend = plt.legend(handles=[line1, line2, line3], loc=1)
        plt.gca().add_artist(legend)
        plt.title("Steps/Reward Loss/Wrong Steps")
        plt.xlabel("Number of Episodes")
        plt.ylabel("Count")
        plt.grid()
        plt.show() 
        
        # Plot policy convergence
        line1, = plt.plot(policy_delta, label = r"Max $\Delta$Q")
        r'i(t) ($\mu$A/$cm^2$)'
        line2, = plt.plot(pd_rolling_mean, label = \
                          r"""Rolling Mean of Max $\Delta$Q
 Window Size = {}""".format(WINDOW_SIZE_POLICY))
        legend2 = plt.legend(handles=[line1, line2], loc=1)
        plt.gca().add_artist(legend2)
        plt.title("Policy Convergence")
        plt.xlabel("Number of Episodes")
        plt.ylabel("Policy Weight")
        plt.grid()
        plt.show()
    
    return env.Q, hunter_train_data

def train_agents(env, hunter, prey, num_episodes):
    """ Trains multiple agents to optimal policy and displays color plots"""
    Q_agents = []
    train_data_agents = []
    
    num_agents = int(input("Input number of agents you would like to train: "))
    if num_agents == 1:
        show_plots = True
    else:
        show_plots = False
    save_data = yes_or_no("Would you like to save the training data?")
    for agent in range(num_agents):
        print("\nTraining agent {}/{}...\n".format(agent + 1, num_agents))
        Q_optimal, hunter_train_data = optimal_policy(env, 
                                                      hunter, 
                                                      prey, 
                                                      num_episodes, 
                                                      show_plots)
        Q_agents.append(Q_optimal)
        train_data_agents.append(hunter_train_data)
        
        # Saving optimal policy agents
        if save_data:
            Q_filename = "Q{}.dat".format(agent + 1)
            hunter_output = "training_data/hunter_data/"
            Q_output = "training_data/Q_tables/"
            np.savetxt(hunter_output + "Hunter_" + Q_filename, 
                       hunter_train_data, 
                       fmt='%-7.3f', 
                       header = "Total Episodes - Hunter Error")
            with open(Q_output + Q_filename, 'w') as outfile:
                outfile.write('# Array shape: {0}\n'.format(Q_optimal.shape))
                for data_slice in Q_optimal:
                    np.savetxt(outfile, data_slice, fmt='%-7.2f')
                    outfile.write('# New slice\n')

    # Show average policy for all directions
    Q_agents_average = np.mean(Q_agents, axis=0)
    Q_agents_average_direction_sum = np.sum(Q_agents_average, axis=2)
    
    # Show average policy for a particular direction
    image_folder = "training_data/images/"
    for direction in range(len(env.action_dict)):
        Q_agents_direction = Q_agents_average[:, :, direction]
        name = [action for action, number in env.action_dict.items() \
                if number == direction]
        name = name[0]
        if save_data:
            show_policy(Q_agents_direction, name, image_folder, save_data)
        else:
            show_policy(Q_agents_direction, name)
    
    if save_data:
        show_policy(Q_agents_average_direction_sum, "every", image_folder, 
                            save_data)
    else:
        show_policy(Q_agents_average_direction_sum, "every")
    
    # Average optimal policy training data
    hunter_train_data_avg = np.mean(train_data_agents, axis=0)
    
    print(
    "\nAverage Total Training Episodes: {}".format(hunter_train_data_avg[0]))
    print(
    "Average Hunter Error Rate: {:.2F}%".format(hunter_train_data_avg[1]*100))
    
    # Saving data for average optimal policy
    if save_data:
        Q_Optimal_filename = "training_data/Q_Optimal.dat"
        hunter_train_data_avg_filename = \
        "training_data/Hunter_Train_Data_Avg.dat" 
        np.savetxt(hunter_train_data_avg_filename, 
                   hunter_train_data_avg, 
                   fmt='%-7.3f', 
                   header = "Average Episodes - Average Hunter Error")
    
        with open(Q_Optimal_filename, 'w') as outfile:
            outfile.write(
                    '# Array shape: {0}\n'.format(Q_agents_average.shape))
            for data_slice in Q_optimal:
                np.savetxt(outfile, data_slice, fmt='%-7.2f')
                outfile.write('# New slice\n')

def rolling_mean(x, N):
    """ Determines rolling mean of x over a given N window"""
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return np.array((cumsum[N:] - cumsum[:-N]) / float(N))

def show_policy(Q_direction, name, image_folder=None, save_data=False):
    """ Displays color plot of policy with given direction (name)"""
    sns.set_palette("Set2")
    plt.figure(figsize=(6, 4))
    plt.imshow(Q_direction, cmap='GnBu', aspect='auto', origin='lower')
    cbar = plt.colorbar()
    cbar.set_label("State Space Value", rotation=90)
    plt.title("Q-table for " + name + " direction")
    plt.xlabel("2 * Hunter Vision + 1 in x")
    plt.ylabel("2 * Hunter Vision + 1 in y")
    if save_data:
        plt.savefig(image_folder + "Q_" + name + ".png", format='png', dpi=500)
    plt.show()

def yes_or_no(question):
    while "The answer is invalid":
        reply = str(input(question+' (y/n): ')).lower().strip()
        if reply[:1] == 'y':
            return True
        if reply[:1] == 'n':
            return False