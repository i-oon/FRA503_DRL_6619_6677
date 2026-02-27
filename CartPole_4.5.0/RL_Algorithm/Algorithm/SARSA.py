from __future__ import annotations
import numpy as np
from RL_Algorithm.RL_base import BaseAlgorithm, ControlType

class SARSA(BaseAlgorithm):
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
        Initialize the SARSA algorithm.
        """
        super().__init__(
            control_type=ControlType.SARSA,
            num_of_action=num_of_action,
            action_range=action_range,
            discretize_state_weight=discretize_state_weight,
            learning_rate=learning_rate,
            initial_epsilon=initial_epsilon,
            epsilon_decay=epsilon_decay,
            final_epsilon=final_epsilon,
            discount_factor=discount_factor,
        )
        
    def update(
        self,
        obs: dict,
        action: int,
        reward: float,
        terminated: bool,
        next_obs: dict,
        next_action: int,  
    ):
        """
        Original single-environment update.
        """
        # Note: discretize_state now returns a list, so we grab index [0]
        state = self.discretize_state(obs)[0]
        next_state = self.discretize_state(next_obs)[0]

        current_q = self.q_values[state][action]

        if terminated:
            next_q = 0.0
        else:
            next_q = self.q_values[next_state][next_action] 

        td_target = reward + (self.discount_factor * next_q)
        td_error = td_target - current_q

        self.q_values[state][action] = current_q + (self.lr * td_error)
        self.training_error.append(td_error)

    def update_batch(
        self, 
        obs: dict, 
        action: np.ndarray, 
        reward: np.ndarray, 
        terminated: np.ndarray, 
        next_obs: dict, 
        next_action: np.ndarray
    ):
        """
        Update the Q-value table using the SARSA Bellman equation for a batch of 256 environments.
        """
        # Convert continuous observations to discrete state tuples for all envs
        states = self.discretize_state(obs)
        next_states = self.discretize_state(next_obs)

        # Loop through all 256 environments and update the Q-table for each one
        for i in range(len(states)):
            state = states[i]
            next_state = next_states[i]
            a = action[i]
            r = reward[i]
            done = terminated[i]
            next_a = next_action[i]

            # Get the current Q-value
            current_q = self.q_values[state][a]

            # Find the Q-value for the ACTUAL next state-action pair
            if done:
                next_q = 0.0
            else:
                next_q = self.q_values[next_state][next_a] 

            # Calculate the Temporal Difference (TD) target
            td_target = r + (self.discount_factor * next_q)

            # Calculate the TD error
            td_error = td_target - current_q

            # Update the Q-value
            self.q_values[state][a] = current_q + (self.lr * td_error)

            # Save the error for later analysis
            self.training_error.append(td_error)



    ### code
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


# Class Agent to implement SARSA reinforcement learning
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
            
            # Tie-breaking if all Q-values are equal
            if mx_nxt_reward == 0 and all(self.Q[(i,j,a)] == 0 for a in self.actions):
                action = np.random.choice(self.actions)
        else:
            action = np.random.choice(self.actions)
        
        position = self.State.nxtPosition(action)
        return position, action
    
    
    # SARSA Algorithm
    def SARSA(self, episodes):
        for x in range(episodes):
            self.State = State()
            self.isEnd = self.State.isEnd
            episode_reward = 0
            
            # Step 1: Initialize S and choose A using epsilon-greedy policy
            i, j = self.State.state
            _, action = self.Action()
            
            while not self.isEnd:
                # Step 2: Take action A, observe R, and get next state S'
                next_state_pos = self.State.nxtPosition(action)
                next_state_obj = State(state=next_state_pos)
                next_state_obj.isEndFunc()
                
                reward = next_state_obj.getReward()
                episode_reward += reward
                
                # Step 3: Choose next action A' from S' using epsilon-greedy policy
                # Temporarily update agent's state to determine next action based on policy
                self.State = next_state_obj
                _, next_action = self.Action()
                
                # Step 4: Update Q-Value using the SARSA formula
                current_q = self.Q[(i, j, action)]
                
                if next_state_obj.isEnd:
                    next_q = 0 # No future rewards if the episode ends
                else:
                    # In SARSA, we use the Q-value of the actual next action we are going to take
                    next_q = self.Q[(next_state_pos[0], next_state_pos[1], next_action)]
                
                # SARSA Update Equation
                new_q = current_q + self.alpha * (reward + self.gamma * next_q - current_q)
                self.Q[(i, j, action)] = round(new_q, 3)
                
                # Step 5: S <- S', A <- A'
                i, j = next_state_pos
                action = next_action
                self.isEnd = next_state_obj.isEnd
                
            self.plot_reward.append(episode_reward)
            
    def plot(self):
        plt.plot(self.plot_reward)
        plt.title('SARSA: Cumulative Reward per Episode')
        plt.xlabel('Episodes')
        plt.ylabel('Reward') k
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
    ag.SARSA(episodes) # Calling the SARSA method instead of Q_Learning
    ag.plot()
    ag.showValues()
    """