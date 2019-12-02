#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  26 08:49:51 2019

@author: Nick Tacca
"""
#%%
import numpy as np

class Environment:
    
    def __init__(self, Nx, Ny):
        
        # Define state space
        self.Nx = Nx
        self.Ny = Ny
        self.state_dim = (Ny, Nx)
        
        # Movement directions
        self.action_dict = {"up": 0, "right": 1, "down": 2, "left": 3}
        self.action_translations = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        
        # Define rewards
        self.target_reward = 100 # Reward for reaching target position
        self.step_penalty = -1 # Penalty for stepping
        
        # Vision of hunter: can see 1 less than max grid dimensions away
        self.hunter_vision = np.max(self.state_dim) - 1
        
        # Agent learning parameters
        self.alpha = 0.1  # Learning rate
        self.gamma = 0.99 # Reward discount factor
        self.tau = 0.01 # Temperature for softmax scaling
        
        # Initialize Q[s,a] table
        self.Q = np.zeros((2 * self.hunter_vision + 1, 
                           2 * self.hunter_vision + 1,
                           len(self.action_dict)))
    
    def step_dist(self, object1, object2):
        step_dist = np.abs(object1.x - object2.x) + \
        np.abs(object1.y - object2.y)
        return step_dist
    
    def total_reward(self, step_dist):
        total_reward = self.target_reward + step_dist * self.step_penalty
        return total_reward
    
    def allowed_actions(self, object):
        # Generate list of actions allowed depending on agent grid location
        actions_allowed = []
        y, x = object.y, object.x
        if (y > 0):  # No passing top edge
            actions_allowed.append(self.action_dict["up"])
        if (y < self.Nx - 1):  # No passing bottom edge
            actions_allowed.append(self.action_dict["down"])
        if (x > 0):  #  No passing left edge
            actions_allowed.append(self.action_dict["left"])
        if (x < self.Ny - 1):  # No passing right edge
            actions_allowed.append(self.action_dict["right"])
        actions_allowed = np.array(actions_allowed, dtype=int)
        return actions_allowed
    
    def give_reward(self, dist):
        # Collect reward
        if dist == 0:
            reward = self.target_reward + self.step_penalty
        else:
            reward = self.step_penalty
        return reward
        
    def softmax(self, Q):
        Q /= self.tau
        Q_max = np.max(Q)
        num = np.exp(Q - Q_max)
        den = np.sum(num)
        prob = num / den
        return prob
    
    def determine_state(self, object1, object2):
        # Determine if target is within agent vision
        x_diff = np.abs(object2.x - object1.x)
        y_diff = np.abs(object2.y - object1.y)
        
        if x_diff <= self.hunter_vision and y_diff <= self.hunter_vision:
            # Target is within sight -> go to target
            sx = (object1.x - object2.x) + self.hunter_vision
            sy = (object1.y - object2.y) + self.hunter_vision
        else:
            # Target not in sight -> explore
            sx = 2 * self.hunter_vision
            sy = 2 * self.hunter_vision
        return sx, sy
                 
    def get_action(self, sx, sy, object):
        actions_allowed = self.allowed_actions(object)
        Q_sa = self.Q[sx, sy, actions_allowed]
        Q_sa_prob = self.softmax(Q_sa)
        return np.random.choice(actions_allowed, p = Q_sa_prob)
    
    def get_experiment_action(self, sx, sy, Q_optimal, error, object):
        actions_allowed = self.allowed_actions(object)
        Q_sa = Q_optimal[sx, sy, actions_allowed]
        Q_sa_prob = self.softmax(Q_sa)
        error = (error/100)
        if np.random.uniform(0, 1) <= error:
            Q_opposite = np.ones(len(Q_sa_prob)) - Q_sa_prob
            return np.random.choice(actions_allowed, 
                                    p = Q_opposite/np.sum(Q_opposite))
        else:
            return np.random.choice(actions_allowed, p = Q_sa_prob)
    
    def train(self, sx, sy, action, reward, sx_, sy_):
        self.Q[sx, sy, action] = (1-self.alpha) * self.Q[sx, sy, action] + \
        self.alpha * (reward + self.gamma * np.max(self.Q[sx_, sy_, :] - \
                                                   self.Q[sx, sy, action]))

class Observation:
    
    def __init__(self, env):
        
        # Defining environment for observations
        self.env = env
        
        # Define observing agent rewards
        self.right_step = 1
        self.wrong_step = -1
        
        # New agent learning parameters
        self.alpha = 0.2  # Learning rate
        self.gamma = 0.5 # Reward discount factor
        self.tau = 0.5 # Temperature for softmax scaling
    
    def give_new_agent_reward(self, sx, sy, Q_optimal, action):
        Q_opt = Q_optimal[sx, sy, :]
        Q_opt_prob = self.softmax(Q_opt)
        actions_ordered = np.argsort(-Q_opt_prob)
        best_action = actions_ordered[0]
        
        # If new agent does best action, give reward
        if action == best_action:
            reward = self.right_step
        else:
            reward = self.wrong_step

        return reward
    
    def determine_error(self, sx, sy, Q_optimal, action):
        # Q error
        Q_opt = np.max(Q_optimal[sx, sy, :])
        Q_opt_sum = np.sum(Q_optimal[sx, sy, :])
        if np.abs(Q_opt_sum) < 0.001:
            Q_opt_sum = 1
        Q_norm = Q_optimal[sx, sy, :]/Q_opt_sum
        #print(Q_norm)
        #Q_agent = Q_optimal[sx, sy, action]
        Q_agent = Q_norm[action]
        
        Q_delta = np.abs(Q_opt - Q_agent)
        
        # Prob delta error
        #Q = Q_optimal[sx, sy, :]
        #Q_prob = self.softmax(Q)
        Q_prob = self.softmax(Q_norm)
        prob_max = np.max(Q_prob)
        prob_action = Q_prob[action]
        
        prob_delta = np.abs(prob_max - prob_action)
        prob_sum = prob_max + prob_action
        prob_delta_norm = np.abs((prob_max - prob_action) / prob_sum)
        
        # Prob error 1-prob
        best_action = np.argmax(Q_prob)
        
        if action == best_action:
            prob_error = 1-prob_max
        else:
            prob_error = 0-prob_max
        
        prob_error = np.abs(prob_error)
#        prob_error = Q_agent
        
        return Q_delta, prob_delta, prob_delta_norm, prob_error
    
    def softmax(self, Q):
        Q /= self.tau
        num = np.exp(Q)
        den = np.sum(num)
        if den == 0:
            den += 0.001
        prob = num / den
        #print(prob)
        return prob

    def train(self, sx, sy, action, reward, sx_, sy_):
            self.env.Q[sx, sy, action] = (1-self.alpha) * \
            self.env.Q[sx, sy, action] + self.alpha * \
            (reward + self.gamma * np.max(self.env.Q[sx_, sy_, :] - \
                                          self.env.Q[sx, sy, action]))

    
class Agent:
    
    def __init__(self, name, x, y, env):
        
        # Name and position of object
        self.name = name
        self.x = x
        self.y = y
        self.env = env
        
    def step(self, action):
        self.x += self.env.action_translations[action][1]
        self.y += self.env.action_translations[action][0]
        
        # Restrict boundaries
        if self.x < 0:
            self.x = 0
        elif self.x > self.env.Ny-1:
            self.x = self.env.Ny-1

        if self.y < 0:
            self.y = 0
        elif self.y > self.env.Nx-1:
            self.y = self.env.Nx-1

class GridworldConstants:
    
    # Grid Dimensions
    NUM_COLS = 5
    NUM_ROWS = 5
    
    # Object Names
    HUNTER_NAME = "Hunter"
    PREY_NAME = "Prey"

def yes_or_no(question):
    while "The answer is invalid":
        reply = str(input(question+' (y/n): ')).lower().strip()
        if reply[:1] == 'y':
            return True
        if reply[:1] == 'n':
            return False
    
# Object Settings
gc = GridworldConstants()
env = Environment(gc.NUM_COLS, gc.NUM_ROWS)
hunter = Agent(gc.HUNTER_NAME, 
               np.random.choice(gc.NUM_COLS), 
               np.random.choice(gc.NUM_ROWS), 
               env)
prey = Agent(gc.PREY_NAME, 
             np.random.choice(gc.NUM_COLS), 
             np.random.choice(gc.NUM_ROWS), 
             env)
obs = Observation(env)