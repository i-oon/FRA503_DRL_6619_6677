from __future__ import annotations
import numpy as np
from RL_Algorithm.RL_base import BaseAlgorithm, ControlType


class Q_Learning(BaseAlgorithm):
    def __init__(
            self,
            num_of_action: int,
            action_range: list,
            discretize_state_weight: list,
            learning_rate: float,
            initial_epsilon: float,
            epsilon_decay: float,
            final_epsilon: float,
            discount_factor: float,
    ) -> None:
        """
        Initialize the Q-Learning algorithm.

        Args:
            num_of_action (int): Number of possible actions.
            action_range (list): Scaling factor for actions.
            discretize_state_weight (list): Scaling factor for discretizing states.
            learning_rate (float): Learning rate for Q-value updates.
            initial_epsilon (float): Initial value for epsilon in epsilon-greedy policy.
            epsilon_decay (float): Rate at which epsilon decays.
            final_epsilon (float): Minimum value for epsilon.
            discount_factor (float): Discount factor for future rewards.
        """
        super().__init__(
            control_type=ControlType.Q_LEARNING,
            num_of_action=num_of_action,
            action_range=action_range,
            discretize_state_weight=discretize_state_weight,
            learning_rate=learning_rate,
            initial_epsilon=initial_epsilon,
            epsilon_decay=epsilon_decay,
            final_epsilon=final_epsilon,
            discount_factor=discount_factor,
        )
        
    def update(self,obs: dict,action: int,reward: float,terminated: bool,next_obs: dict,):
        """
        Update the Q-value table using the Q-learning Bellman equation.
        """
        # 1. Convert continuous observations to discrete state tuples
        if hasattr(self, '_update_count'):
            self._update_count += 1
        else:
            self._update_count = 1
    
        state = self.discretize_state(obs)
        next_state = self.discretize_state(next_obs)

        # 2. Get the current Q-value for the state-action pair
        # Note: self.q_values is a dict of numpy arrays, so we index the state, then the action.
        current_q = self.q_values[state][action]

        # 3. Find max Q-value for the next state
        if terminated:
            max_next_q = 0.0  # No future rewards if the episode is over
        else:
            max_next_q = np.max(self.q_values[next_state])

        # 4. Calculate the Temporal Difference (TD) target
        td_target = reward + (self.discount_factor * max_next_q)

        # 5. Calculate the TD error
        td_error = td_target - current_q

        # 6. Update the Q-value (keeping exact float values, no rounding)
        self.q_values[state][action] = current_q + (self.lr * td_error)
        if self._update_count <= 10:
            print(f"Update #{self._update_count}:")
            print(f"  State: {state}, Action: {action}")
            print(f"  Reward: {reward:.4f}, Terminated: {terminated}")
            print(f"  Current Q: {current_q:.4f} -> New Q: {self.q_values[state][action]:.4f}")
            print(f"  TD Error: {td_error:.4f}\n")
        # 7. Save the error for later analysis (optional but helpful)
        self.training_error.append(td_error)


"""
# -*- coding: utf-8 -*-

import numpy as np
import random
import matplotlib.pyplot as plt

# Set the rows and columns length
BOARD_ROWS = 5
BOARD_COLS = 5

# Initialise start, win and lose states
START = (0, 0)
WIN_STATE = (4, 4)
HOLE_STATE = [(1,0), (3,1), (4,2), (1,3)]

# Class State defines the board and decides reward, end and next position
class State:
    def __init__(self, state=START):
        # Initialise the state to start and end to false
        self.state = state
        self.isEnd = False
        self.isEndFunc() # Check if the initial state is already an end state

    def getReward(self):
        # Give the rewards for each state -5 for loss, +1 for win, -1 for others
        if self.state in HOLE_STATE:
            return -5
        elif self.state == WIN_STATE:
            return 1       
        else:
            return -1

    def isEndFunc(self):
        # Set state to end if win/loss
        if self.state == WIN_STATE:
            self.isEnd = True
            
        if self.state in HOLE_STATE:
            self.isEnd = True

    def nxtPosition(self, action):     
        # Set the positions from current action - up, down, left, right
        if action == 0:                
            nxtState = (self.state[0] - 1, self.state[1]) # up             
        elif action == 1:
            nxtState = (self.state[0] + 1, self.state[1]) # down
        elif action == 2:
            nxtState = (self.state[0], self.state[1] - 1) # left
        else:
            nxtState = (self.state[0], self.state[1] + 1) # right

        # Check if next state is possible (within the grid boundaries)
        if (nxtState[0] >= 0) and (nxtState[0] <= 4):
            if (nxtState[1] >= 0) and (nxtState[1] <= 4):    
                return nxtState 
                
        # Return current state if outside grid     
        return self.state 


# Class Agent to implement reinforcement learning through grid  
class Agent:
    def __init__(self):
        # Initialise states and actions 
        self.actions = [0, 1, 2, 3]    # up, down, left, right
        self.State = State()
        
        # Set the learning and greedy values
        self.alpha = 0.5
        self.gamma = 0.9
        self.epsilon = 0.1
        self.isEnd = self.State.isEnd

        # Array to retain reward values for plot
        self.plot_reward = []
        
        # Initialise Q values as a dictionary
        self.Q = {}
        self.rewards = 0
        
        # Initialise all Q values across the board to 0
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                for k in range(len(self.actions)):
                    self.Q[(i, j, k)] = 0
        

    # Method to choose action with Epsilon greedy policy
    def Action(self):
        # Random value vs epsilon
        rnd = random.random()
        mx_nxt_reward = -float('inf')
        action = None
        
        # 90% of the time find max Q value over actions (Exploitation)
        if rnd > self.epsilon:
            for k in self.actions:
                i, j = self.State.state
                nxt_reward = self.Q[(i, j, k)]
                
                if nxt_reward > mx_nxt_reward:
                    action = k
                    mx_nxt_reward = nxt_reward
            # Handle tie-breaking randomly if all Q-values are equal
            if mx_nxt_reward == 0 and all(self.Q[(i,j,a)] == 0 for a in self.actions):
                action = np.random.choice(self.actions)
                    
        # 10% of the time choose random action (Exploration)
        else:
            action = np.random.choice(self.actions)
        
        # Select the next state based on action chosen
        position = self.State.nxtPosition(action)
        return position, action
    
    
    # Q-learning Algorithm
    def Q_Learning(self, episodes):
        for x in range(episodes):
            # Reset state for the new episode
            self.State = State()
            self.isEnd = self.State.isEnd
            episode_reward = 0
            
            # Continue until reaching terminal state (Win or Hole)
            while not self.isEnd:
                i, j = self.State.state
                next_state_pos, action = self.Action()
                
                # Create next state object
                next_state_obj = State(state=next_state_pos)
                next_state_obj.isEndFunc()
                
                # Get reward of the NEXT state
                reward = next_state_obj.getReward()
                episode_reward += reward
                
                # Find max Q value for the next state
                if next_state_obj.isEnd:
                    mx_nxt_value = 0 # No future rewards if the episode ends
                else:
                    mx_nxt_value = max([self.Q[(next_state_pos[0], next_state_pos[1], a)] for a in self.actions])
                
                # Update Q-value (In-place)
                current_q = self.Q[(i, j, action)]
                new_q = current_q + self.alpha * (reward + self.gamma * mx_nxt_value - current_q)
                self.Q[(i, j, action)] = round(new_q, 3)
                
                # Move to the next state
                self.State = next_state_obj
                self.isEnd = self.State.isEnd
                
            # Append total reward of the episode for plotting
            self.plot_reward.append(episode_reward)
            
    # Plot the reward vs episodes
    def plot(self):
        plt.plot(self.plot_reward)
        plt.title('Cumulative Reward per Episode')
        plt.xlabel('Episodes')
        plt.ylabel('Reward')
        plt.show()
        
        
    # Iterate through the board and find largest Q value in each, print output
    def showValues(self):
        print("\nMax Q-Values for each state (Grid):")
        for i in range(0, BOARD_ROWS):
            print('---------------------------------------------------------')
            out = '| '
            for j in range(0, BOARD_COLS):
                mx_nxt_value = -float('inf')
                for a in self.actions:
                    nxt_value = self.Q[(i, j, a)]
                    if nxt_value > mx_nxt_value:
                        mx_nxt_value = nxt_value
                
                # Format output to look like a grid
                if (i, j) == WIN_STATE:
                    out += " WIN  | "
                elif (i, j) in HOLE_STATE:
                    out += " HOLE | "
                else:
                    out += str(round(mx_nxt_value, 2)).ljust(5) + ' | '
            print(out)
        print('---------------------------------------------------------')
        
        
if __name__ == "__main__":
    # Create agent for 10,000 episodes implementing a Q-learning algorithm
    ag = Agent()
    episodes = 10000
    ag.Q_Learning(episodes)
    ag.plot()
    ag.showValues()
"""