import gym 
from tensorflow import keras 
import random
import numpy as np
from collections import deque
from keras.utils.vis_utils import plot_model


class ddq_net:


	def __init__(self, state_size, action_size ):
		self.state_size = state_size
		self.action_size = action_size
		
		self.memory = deque(maxlen=2000)
		
		self.epsilon = 1
		self.epsilon_decay = 0.995
		self.epsilon_min = 0.001
		self.gamma = 0.99
		
		self.batch_size = 32
		
		self.model = self.build_model()
	
	def build_model(self):
	
		model = keras.models.Sequential()
		
		model.add(keras.layers.Dense(24, input_shape=(self.state_size,), activation='relu'))
		model.add(keras.layers.Dense(24, activation='relu'))
		model.add(keras.layers.Dense(self.action_size,activation='linear'))
		
		model.compile(loss='mse', optimizer='adam')
		
		return model
		
	def remember(self,state,action,reward,new_state,done):
		self.memory.append((state,action,reward,new_state,done))
		
	def replay(self, batch_size):
	
		mini_batch = random.sample(self.memory, batch_size)
		
		for state, action, reward, new_state, done in mini_batch:
		
			target = reward
			
			if not done:
				target = reward + self.gamma * np.max( self.model.predict(new_state)[0])
			
			target_q = self.model.predict(state)
			target_q[0][action] = target
			
			self.model.fit(state,target_q,epochs=1, verbose=0)

		if self.epsilon > self.epsilon_min:
			self.epsilon *= self.epsilon_decay

	def act(self,state):
	
		if np.random.rand() >= self.epsilon:
			return random.randrange(self.action_size)
		
		action = self.model.predict(state)
		return np.argmax(action[0])
	
def train_model():

	env = gym.make('CartPole-v1')
	
	state_size, action_size = env.observation_space.shape[0], env.action_space.n
	
	agent = ddq_net(state_size, action_size)
	
	episodes = 1000
		
	for episode in range(episodes):
	
		done = False
		steps = 0
		state = env.reset()
		
		state = np.reshape(state, [1,state_size])
		
		while not done:
		
			env.render()
			
			steps += 1
			
			action = agent.act(state)
			
			new_state, reward, done, _ = env.step(action)
			
			new_state = np.reshape(new_state, [1,state_size])
			
			agent.remember(state,action,reward,new_state,done)
		
			state = new_state
		
				
		agent.replay(32 if steps > 32 else steps)
		
		
	env.close()
			
train_model()
		
		
		
		
		
		
		
		
		
			
	

