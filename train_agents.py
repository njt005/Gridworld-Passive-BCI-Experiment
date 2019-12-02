#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 11:18:44 2019

@author: nicktacca
"""
from gridworld_settings import env, hunter, prey
from training_settings import train_agents

# Maximum number of observation training episodes
NUM_TRAIN_EPISODES = 10000

# Begin program
if __name__ == "__main__":
   train_agents(env, hunter, prey, NUM_TRAIN_EPISODES)