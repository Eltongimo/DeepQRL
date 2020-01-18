# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 21:44:26 2019

@author: EltonUG
"""

import numpy as np

class Grid_world:
    
    # the players position is reprensented by 1 
    def __init__(self,row,colum):
        self.row = row
        self.colum = colum
        self.target_position = (row-1,colum-1)
        self.grid = np.zeros((row,colum))
        self.rewards = np.zeros(self.grid.shape)
        self.rewards = self.fit_rewards_table()
        self.position = (0,0) 
        self.grid[self.position] = 1 # initializing the pleyer at random position
        self.grid[self.target_position] = 200
        self.action_space = 4
        self.state_space = row 
        self.state_mapping_dict = self.fit_state_mapping()
      
    def randomize_position(self):
        return (np.random.randint(0,self.row - 1), np.random.randint(0,self.colum - 1))
    
    def random_action(self):
        return np.random.randint(0,4)
    
    def fit_state_mapping(self):
        
        state = 0 
        d = {}
        
        for i in range(self.row):
            for j in range(self.colum):
                d[(i,j)] = state
                state += 1 
        return d
     
    def fit_rewards_table(self):

        
        for i in range(self.row):
            for j in range(self.colum):
                self.rewards[i,j] = -1
        
        self.rewards[self.target_position] = 1000 
        return self.rewards    
    
    
    def is_valid_position(self,action):
        
        pos = self.get_position(action)
          
        row , column = zip(pos)
        
        if row[0] < 0 or row[0] >= self.row or column[0] < 0 or column[0] >= self.colum:
            return False
        return True
        
    
    def set_environment(self,position,value):
        self.grid[position] = value
        
    def get_position(self,action):
        
        r,c = zip(self.position)
        r = r[0]
        c = c[0]
        
        if action == 0:
            c -= 1
        elif action == 1:
            c += 1
        elif action == 2:
            r -= 1
        elif action == 3:
            r += 1
        return (r,c)
    
    def step(self, action):
        
        if  self.is_valid_position(action):
                
            self.set_environment(self.position,0)
            
            self.position = self.get_position(action)
            
            self.set_environment(self.position,1)
        
        # return self.state_mapping_dict[self.position],self.rewards[self.position], self.position == self.target_position
		# this return changed because i was testing the state as a matrix of [row,col] as input to neural network in script
		# DQN_GrigWorld.py
		
        return self.grid,self.rewards[self.position], self.position == self.target_position
    
	
    def reset(self):
        self.position = self.randomize_position()
        self.grid = np.zeros(self.grid.shape)
        self.grid[self.position] = 1
        #return self.state_mapping_dict[self.position]
        return self.grid
		
def Qlearning_agent():
    row = 10
    col = 10
    
    agent = Grid_world(row,col)
    
    done = False
    GAMMA = 0.1
    ALPHA = 0.5
    EPISODES = 200000
    EPSILON = 0.995
    EPSILON_DECAY = 0.99
    
    Q_table = np.zeros((agent.state_space,agent.action_space))
    
   
    for episode in range(EPISODES):
        
        done = False
        state = agent.reset()
        steps = 0
        
        while not done:
            steps += 1
           
            action = np.argmax(Q_table[state])
        
            new_state,reward,done = agent.step(action)
            
            
            Q_table[state,action] = Q_table[state,action] + ALPHA *( reward + GAMMA * np.max(Q_table[new_state])\
                       - Q_table[state,action])
            
            state = new_state
            
        EPSILON *= EPSILON_DECAY
          
        print("Episode {} with timesteps {} ".format(episode,steps))
    
if __name__ == '__main__':
    
    Qlearning_agent()
    

    
    
            
            

            
        
        
        
            
        
        
