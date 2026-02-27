from __future__ import annotations
import numpy as np
from collections import defaultdict
from RL_Algorithm.RL_base import BaseAlgorithm, ControlType

class MC(BaseAlgorithm):
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
        Initialize the Monte Carlo algorithm.
        """
        super().__init__(
            control_type=ControlType.MONTE_CARLO,
            num_of_action=num_of_action,
            action_range=action_range,
            discretize_state_weight=discretize_state_weight,
            learning_rate=learning_rate,
            initial_epsilon=initial_epsilon,
            epsilon_decay=epsilon_decay,
            final_epsilon=final_epsilon,
            discount_factor=discount_factor,
        )
        
        # Create separate memory buffers for all 256 environments
        self.obs_hist_dict = defaultdict(list)
        self.action_hist_dict = defaultdict(list)
        self.reward_hist_dict = defaultdict(list)

    def update(self): 
        """
        Original single-environment update (kept for compatibility).
        """
        G = 0.0  
        for state, action, reward in zip(reversed(self.obs_hist), reversed(self.action_hist), reversed(self.reward_hist)):
            G = self.discount_factor * G + reward  
            state_dis = self.discretize_state(state)[0]
            current_q = self.q_values[state_dis][action]  
            td_error = G - current_q
            self.q_values[state_dis][action] = current_q + (self.lr * td_error)  
            self.training_error.append(td_error)

        self.obs_hist.clear()
        self.action_hist.clear()
        self.reward_hist.clear()

    def _update_single_env(self, env_idx: int):
        """
        Helper function to calculate G and update Q-table for ONE specific environment
        that just finished its episode.
        """
        G = 0.0
        obs_hist = self.obs_hist_dict[env_idx]
        action_hist = self.action_hist_dict[env_idx]
        reward_hist = self.reward_hist_dict[env_idx]
        
        # Check if trajectory is empty
        if len(obs_hist) == 0:
            return
        
        # Iterate backwards through this specific environment's trajectory
        for state_dis, action, reward in zip(reversed(obs_hist), reversed(action_hist), reversed(reward_hist)):
            G = self.discount_factor * G + reward
            
            current_q = self.q_values[state_dis][action]
            td_error = G - current_q
            self.q_values[state_dis][action] = current_q + (self.lr * td_error)
            self.training_error.append(td_error)
            
        # Clear the buffers for this environment so it can start fresh
        self.obs_hist_dict[env_idx].clear()
        self.action_hist_dict[env_idx].clear()
        self.reward_hist_dict[env_idx].clear()

    def update_batch(self, obs: dict, action: np.ndarray, reward, terminated, next_obs: dict):
        """
        Gather experience for all 256 environments, and trigger updates for any that finish.
        """
        # Discretize all 256 current states
        states = self.discretize_state(obs)

        # Loop through the batch
        for i in range(len(states)):
            # Handle action indexing (might be array or scalar)
            action_idx = action[i] if hasattr(action, '__len__') else action
            
            # Record the step memory for environment `i`
            self.obs_hist_dict[i].append(states[i])
            self.action_hist_dict[i].append(action_idx)
            self.reward_hist_dict[i].append(reward[i].item() if hasattr(reward[i], 'item') else reward[i])
            
            # If environment `i` just died/finished, calculate its full return and update!
            terminated_value = terminated[i].item() if hasattr(terminated[i], 'item') else terminated[i]
            if terminated_value:
                self._update_single_env(i)




"""
    # -*- coding: utf-8 -*-
"""
"""

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
        self.state = state
        self.isEnd = False
        self.isEndFunc()

    def getReward(self):
        if self.state in HOLE_STATE:
            return -5
        elif self.state == WIN_STATE:
            return 1       
        else:
            return -1

    def isEndFunc(self):
        if self.state == WIN_STATE:
            self.isEnd = True
        if self.state in HOLE_STATE:
            self.isEnd = True

    def nxtPosition(self, action):     
        if action == 0:                
            nxtState = (self.state[0] - 1, self.state[1]) # up             
        elif action == 1:
            nxtState = (self.state[0] + 1, self.state[1]) # down
        elif action == 2:
            nxtState = (self.state[0], self.state[1] - 1) # left
        else:
            nxtState = (self.state[0], self.state[1] + 1) # right

        if (nxtState[0] >= 0) and (nxtState[0] <= 4):
            if (nxtState[1] >= 0) and (nxtState[1] <= 4):    
                return nxtState 
                
        return self.state 


# Class Agent to implement Monte Carlo learning
class Agent:
    def __init__(self):
        self.actions = [0, 1, 2, 3]    # up, down, left, right
        self.State = State()
        
        self.alpha = 0.5
        self.gamma = 0.9
        self.epsilon = 0.1
        self.isEnd = self.State.isEnd

        self.plot_reward = []
        self.Q = {}
        
        # Initialise all Q values across the board to 0
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                for k in range(len(self.actions)):
                    self.Q[(i, j, k)] = 0
        
    def Action(self):
        rnd = random.random()
        mx_nxt_reward = -float('inf')
        action = None
        
        if rnd > self.epsilon:
            for k in self.actions:
                i, j = self.State.state
                nxt_reward = self.Q[(i, j, k)]
                
                if nxt_reward > mx_nxt_reward:
                    action = k
                    mx_nxt_reward = nxt_reward
            
            # Tie-breaking
            if mx_nxt_reward == 0 and all(self.Q[(i,j,a)] == 0 for a in self.actions):
                action = np.random.choice(self.actions)
        else:
            action = np.random.choice(self.actions)
        
        position = self.State.nxtPosition(action)
        return position, action
    
    
    # Monte Carlo Algorithm (Every-Visit MC Control)
    def Monte_Carlo(self, episodes):
        for x in range(episodes):
            self.State = State()
            self.isEnd = self.State.isEnd
            
            # Array to store the trajectory of the current episode: (State, Action, Reward)
            trajectory = []
            
            # --- Phase 1: Generate an entire episode ---
            while not self.isEnd:
                i, j = self.State.state
                next_state_pos, action = self.Action()
                
                next_state_obj = State(state=next_state_pos)
                next_state_obj.isEndFunc()
                reward = next_state_obj.getReward()
                
                # Record the step
                trajectory.append(((i, j), action, reward))
                
                self.State = next_state_obj
                self.isEnd = self.State.isEnd
                
            # --- Phase 2: Learn from the episode at the end ---
            G = 0  # G is the cumulative discounted reward (Return)
            episode_total_reward = sum([step[2] for step in trajectory]) # For plotting
            self.plot_reward.append(episode_total_reward)
            
            # Iterate backwards through the trajectory to calculate G efficiently
            for state, action, reward in reversed(trajectory):
                # Calculate discounted return G
                G = self.gamma * G + reward
                
                # Update Q-Value for the state-action pair using MC formula
                current_q = self.Q[(state[0], state[1], action)]
                new_q = current_q + self.alpha * (G - current_q)
                self.Q[(state[0], state[1], action)] = round(new_q, 3)
            
    def plot(self):
        plt.plot(self.plot_reward)
        plt.title('Monte Carlo: Cumulative Reward per Episode')
        plt.xlabel('Episodes')
        plt.ylabel('Reward')
        plt.show()
        
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
                
                if (i, j) == WIN_STATE:
                    out += " WIN  | "
                elif (i, j) in HOLE_STATE:
                    out += " HOLE | "
                else:
                    out += str(round(mx_nxt_value, 2)).ljust(5) + ' | '
            print(out)
        print('---------------------------------------------------------')
        
        
if __name__ == "__main__":
    ag = Agent()
    episodes = 10000
    ag.Monte_Carlo(episodes) # Calling the Monte Carlo method
    ag.plot()
    ag.showValues()
    """