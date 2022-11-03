import numpy as np
from gym import Env
from gym.spaces import Box, Discrete, Tuple
import random
from tqdm import tqdm


class CustomEnv(Env):
  def __init__(self):
    self.action_space = Discrete(5)
    #self.action_space = Box(low=np.array([-10, -10]), high=np.array([10, 10]))
    self.observation_space = Box(low=np.array([-100]), high=np.array([100]))
    self.state = [ 38 + random.randint(-3,3),  38 ]
    self.shower_length = 500
  
  def step(self, action):
    #print(action)
    self.shower_length -= 1 
    
    if action == 0:
      pass
    elif action == 1:
      self.state[0] -= 1
    elif action == 2:
      self.state[1] -= 1
    elif action == 3:
      self.state[0] += 1
    elif action == 4:
      self.state[1] += 1
    
    # Calculating the reward
    #self.state += action
    afterState = -1000 * ( np.abs(self.state[0] - 30) + np.abs(self.state[1] - 50) )
    
    #print(self.state)
    reward = afterState

    # Checking if shower is done
    if self.shower_length <= 0:
      done = True
    else:
      done = False
    
    # Setting the placeholder for info
    info = {}
    
    # Returning the step information
    return self.state, reward, done, info
  
  def render(self):
    # This is where you would write the visualization code
    pass

  def reset(self):
    self.state = [ 100 * int( ( random.random()-0.5) ), 100*int( (random.random()-0.5) ) ]
    #self.state = [ 50 , 100 ]
    self.shower_length = 500
    return self.state

env = CustomEnv()
scoreSet = []
stateSet = []

episodes = 100000 #20 shower episodes
tq = tqdm(range(episodes), ncols=40)

for episode in tq:
  state = env.reset()
  done = False
  score = 0 
  
  while not done:
    action = env.action_space.sample()
    n_state, reward, done, info = env.step(action)
    score+=reward
  
  scoreSet.append(reward)
  stateSet.append(env.state)
  #print('Episode:{} Score:{}'.format(episode, score))
  if (episode+1) % 100 == 0:
    
    stateSet[ np.argmax(scoreSet) ], max(scoreSet)