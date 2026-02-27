from __future__ import annotations
import numpy as np
from RL_Algorithm.RL_base import BaseAlgorithm, ControlType

class Double_Q_Learning(BaseAlgorithm):
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
        Initialize the Double Q-Learning algorithm.
        """
        super().__init__(
            control_type=ControlType.DOUBLE_Q_LEARNING,
            num_of_action=num_of_action,
            action_range=action_range,
            discretize_state_weight=discretize_state_weight,
            learning_rate=learning_rate,
            initial_epsilon=initial_epsilon,
            epsilon_decay=epsilon_decay,
            final_epsilon=final_epsilon,
            discount_factor=discount_factor,
        )
        
    def update(self, obs: dict, action: int, reward: float, terminated: bool, next_obs: dict):
        """
        Original single-environment update.
        """
        # Note: discretize_state now returns a list, so we grab index [0]
        state_dis = self.discretize_state(obs)[0]
        next_state_dis = self.discretize_state(next_obs)[0]
        
        # Select a random Q-table (Q1 or Q2) to update
        if np.random.random() < 0.5:
            # --- Update Q1 (qa_values) ---
            best_next_action_q1 = int(np.argmax(self.qa_values[next_state_dis]))
            eval_q2 = self.qb_values[next_state_dis][best_next_action_q1] if not terminated else 0.0
            current_q1 = self.qa_values[state_dis][action]
            
            # Update Q1
            td_error = reward + self.discount_factor * eval_q2 - current_q1
            self.qa_values[state_dis][action] += self.lr * td_error
            self.training_error.append(td_error)
            
        else:
            # --- Update Q2 (qb_values) ---
            best_next_action_q2 = int(np.argmax(self.qb_values[next_state_dis]))
            eval_q1 = self.qa_values[next_state_dis][best_next_action_q2] if not terminated else 0.0
            current_q2 = self.qb_values[state_dis][action]
            
            # Update Q2
            td_error = reward + self.discount_factor * eval_q1 - current_q2
            self.qb_values[state_dis][action] += self.lr * td_error
            self.training_error.append(td_error)

        # Average them for the main Q-table used in action selection
        self.q_values[state_dis][action] = (self.qa_values[state_dis][action] + self.qb_values[state_dis][action]) / 2.0

    def update_batch(self, obs: dict, action: np.ndarray, reward: np.ndarray, terminated: np.ndarray, next_obs: dict):
        """
        Update Q-values using Double Q-Learning for a batch of 256 environments simultaneously.
        """
        states = self.discretize_state(obs)
        next_states = self.discretize_state(next_obs)

        # Loop through all 256 environments
        for i in range(len(states)):
            state = states[i]
            next_state = next_states[i]
            a = action[i]
            r = reward[i]
            done = terminated[i]

            # Select a random Q-table (Q1 or Q2) to update for this specific environment
            if np.random.random() < 0.5:
                # --- Update Q1 (qa_values) ---
                if done:
                    eval_q2 = 0.0
                else:
                    best_next_action_q1 = int(np.argmax(self.qa_values[next_state]))
                    eval_q2 = self.qb_values[next_state][best_next_action_q1]
                
                current_q1 = self.qa_values[state][a]
                td_error = r + (self.discount_factor * eval_q2) - current_q1
                self.qa_values[state][a] += self.lr * td_error
                self.training_error.append(td_error)
                
            else:
                # --- Update Q2 (qb_values) ---
                if done:
                    eval_q1 = 0.0
                else:
                    best_next_action_q2 = int(np.argmax(self.qb_values[next_state]))
                    eval_q1 = self.qa_values[next_state][best_next_action_q2]
                
                current_q2 = self.qb_values[state][a]
                td_error = r + (self.discount_factor * eval_q1) - current_q2
                self.qb_values[state][a] += self.lr * td_error
                self.training_error.append(td_error)

            # Keep the main Q-table updated as the average of QA and QB
            self.q_values[state][a] = (self.qa_values[state][a] + self.qb_values[state][a]) / 2.0

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


# Class Agent to implement Double Q-learning
class Agent:
    def __init__(self):
        self.actions = [0, 1, 2, 3]    # up, down, left, right
        self.State = State()
        
        self.alpha = 0.5
        self.gamma = 0.9
        self.epsilon = 0.1
        self.isEnd = self.State.isEnd

        self.plot_reward = []
        
        # Initialize TWO Q-tables to 0
        self.Q1 = {}
        self.Q2 = {}
        
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                for k in range(len(self.actions)):
                    self.Q1[(i, j, k)] = 0
                    self.Q2[(i, j, k)] = 0
        
    def Action(self):
        rnd = random.random()
        mx_nxt_reward = -float('inf')
        action = None
        
        # We use the SUM of Q1 and Q2 to determine the best action for our epsilon-greedy policy
        if rnd > self.epsilon:
            for k in self.actions:
                i, j = self.State.state
                # Combine Q1 and Q2 for action selection
                nxt_reward = self.Q1[(i, j, k)] + self.Q2[(i, j, k)]
                
                if nxt_reward > mx_nxt_reward:
                    action = k
                    mx_nxt_reward = nxt_reward
            
            # Tie-breaking if all values are equal
            if mx_nxt_reward == 0 and all((self.Q1[(i,j,a)] + self.Q2[(i,j,a)]) == 0 for a in self.actions):
                action = np.random.choice(self.actions)
        else:
            action = np.random.choice(self.actions)
        
        position = self.State.nxtPosition(action)
        return position, action
    
    
    # Double Q-learning Algorithm
    def Double_Q_Learning(self, episodes):
        for x in range(episodes):
            self.State = State()
            self.isEnd = self.State.isEnd
            episode_reward = 0
            
            while not self.isEnd:
                i, j = self.State.state
                next_state_pos, action = self.Action()
                
                next_state_obj = State(state=next_state_pos)
                next_state_obj.isEndFunc()
                reward = next_state_obj.getReward()
                episode_reward += reward
                
                # 50/50 chance to update either Q1 or Q2
                if random.random() < 0.5:
                    # --- UPDATE Q1 ---
                    if next_state_obj.isEnd:
                        eval_q2 = 0
                    else:
                        # Find the BEST action in the next state according to Q1
                        best_next_action_q1 = max(self.actions, key=lambda a: self.Q1[(next_state_pos[0], next_state_pos[1], a)])
                        # Evaluate that best action using Q2
                        eval_q2 = self.Q2[(next_state_pos[0], next_state_pos[1], best_next_action_q1)]
                        
                    current_q1 = self.Q1[(i, j, action)]
                    new_q1 = current_q1 + self.alpha * (reward + self.gamma * eval_q2 - current_q1)
                    self.Q1[(i, j, action)] = round(new_q1, 3)
                    
                else:
                    # --- UPDATE Q2 ---
                    if next_state_obj.isEnd:
                        eval_q1 = 0
                    else:
                        # Find the BEST action in the next state according to Q2
                        best_next_action_q2 = max(self.actions, key=lambda a: self.Q2[(next_state_pos[0], next_state_pos[1], a)])
                        # Evaluate that best action using Q1
                        eval_q1 = self.Q1[(next_state_pos[0], next_state_pos[1], best_next_action_q2)]
                        
                    current_q2 = self.Q2[(i, j, action)]
                    new_q2 = current_q2 + self.alpha * (reward + self.gamma * eval_q1 - current_q2)
                    self.Q2[(i, j, action)] = round(new_q2, 3)
                
                self.State = next_state_obj
                self.isEnd = self.State.isEnd
                
            self.plot_reward.append(episode_reward)
            
    def plot(self):
        plt.plot(self.plot_reward)
        plt.title('Double Q-Learning: Cumulative Reward per Episode')
        plt.xlabel('Episodes')
        plt.ylabel('Reward')
        plt.show()
        
    def showValues(self):
        print("\nMax Combined Q-Values (Q1 + Q2) / 2 for each state:")
        for i in range(0, BOARD_ROWS):
            print('---------------------------------------------------------')
            out = '| '
            for j in range(0, BOARD_COLS):
                mx_nxt_value = -float('inf')
                for a in self.actions:
                    # Average the two Q-tables for final display
                    nxt_value = (self.Q1[(i, j, a)] + self.Q2[(i, j, a)]) / 2.0
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
    ag.Double_Q_Learning(episodes) # Call Double Q-learning
    ag.plot()
    ag.showValues()
    """