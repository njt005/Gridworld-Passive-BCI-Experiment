#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 11:18:44 2019

@author: nicktacca
"""
#%%
from gridworld_settings import env, obs, hunter, prey
from observation_settings import multi_agents_observation

# Begin program
if __name__ == "__main__":
   multi_agents_observation(env, obs, hunter, prey)