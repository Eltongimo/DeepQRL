import numpy as np
import gym
import keras 
import random 
from collections import deque
from keras import backend as K 
import tensorflow as tf

class ddqn_net:

	def __init__(self, state_size, action_size):
	
		# useful for define the ann Architeture
		self.state_size = state_size
		self.action_size = action_size
		
		#memory experinces 
		self.memory = deque(maxlen=2000)
		
		# defining the hyperparameters 
		self.gamma = 0.95
		self.epsilon = 1.0 # exploration parameter
		self.epsilon_decay = 0.99 # exploration decreasing rating
		self.epsilon_min = 0.001 # minimum exploration point
		
		#defining the model 
		self.model = self.build_model()
		self.target_model = self.build_model()
		self.update_target_model()
		
	def _huber_loss(self,y_true,y_pred,clip_delta=1.0):
		error = y_true - y_pred
		cond  = K.abs(error) <= clip_delta
		squared_loss = 0.5 * K.square(error)
		quadratic_loss = 0.5 * K.square(clip_delta) + clip_delta * (K.abs(error) - clip_delta)
		return K.mean(tf.where(cond, squared_loss, quadratic_loss))

	def replay(self,batch_size):
	 
		# sample a mini batch experience 
		mini_batch = random.sample(self.memory, batch_size)
		
		for state, action, reward, new_state, done in mini_batch:
		
			target = self.model.predict(state)
			
			if done:
				target[0][action] = reward
			else:
				t = self.target_model.predict(new_state)
				target[0][action] = reward + self.gamma * np.max(t[0])
			
			self.model.fit(state, target, epochs=1, verbose=0 )
		
		if self.epsilon > self.epsilon_min:
			self.epsilon *= self.epsilon_decay
	
	def update_target_model(self):
		# copying the weights from model to the target model
		self.target_model.set_weights(self.model.get_weights())
	
	def remember(self, state, action, reward , next_state, done):
		self.memory.append((state,action,reward,next_state,done))
	
	def build_model(self):
		# defining the deep neural network structure
		model = keras.models.Sequential()
		model.add(keras.layers.Dense(24,input_shape=(self.state_size,),activation='relu'))
		model.add(keras.layers.Dense(24,activation='relu'))
		model.add(keras.layers.Dense(self.action_size,activation='linear'))
		model.compile(loss=self._huber_loss, optimizer=keras.optimizers.Adam(lr=0.0025))
		
		return model
		
	def act(self, state):
		# choose action based on explore or exploit policy
		if np.random.rand() <= self.epsilon:
			return random.randrange(self.action_size)
		action = self.model.predict(state)
		
		return np.argmax(action[0])
		
	def save(self, name):
		self.model.save_weights(name)
	
	def load(self,name):
		self.model.load_weights(name)

def train_model():

	env = gym.make('CartPole-v0')
	state_space, action_space = env.observation_space.shape[0], env.action_space.n
	
	agent = ddqn_net(state_space, action_space)
	
	batch_size = 32
	
	episodes = 10000
	
	for episode in range(episodes):
	
		done = False
		state = env.reset()
		state = np.reshape(state, [1, state_space])
		
		
		time = 0
		while not done:
			time += 1
			
			action = agent.act(state)
			
			next_state, reward, done, info = env.step(action)
			
			reward = reward if done else -10
			
			next_state = np.reshape(next_state, [1,state_space])
			
			agent.remember(state, action, reward , next_state, done)
			
			state = next_state
			
			if len(agent.memory) > batch_size:
				agent.replay(batch_size)
				
			if done:
				agent.update_target_model()
				print (f"episode {episode} -- time {time} --epsilon {agent.epsilon}")
				break
				
				
	agent.save('./weights/weights.h5')
	env.close()		
			

train_model()			
				
			
			
	
		
		
	
		
		
		
		