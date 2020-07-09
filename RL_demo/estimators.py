import numpy as np
from vowpalwabbit import pyvw
from collections import deque 

class Evaluation:
    def __init__(self, horizon, num_target_policies, num_actions, target_policies):
        """
        inputs
        :param horizon: size of horizon (number of requests)
        :param num_target_policies: number of target policies
        :param num_actions: number of actions avilable (number of servers)
        """
        self.horizon = horizon
        self.num_target_policies = num_target_policies
        self.num_actions = num_actions
        self.num_trajectories = 0 # number of trajectories received so far
        self.target_policies = target_policies

        self.rewards = [] # Holds the per-step rewards for each trajectory
        self.per_trajectory_cumulative_importance_ratios = [] # Holds the per-step importance ratios of each action for each policy for each trajectory

        # For the DR estimator
        self.DR_estimates = [[] for _ in range(self.num_target_policies)]
        self.models = [[pyvw.vw("--power_t 0.0 -q ca --quiet") for _ in range(self.horizon)] for _ in range(self.num_target_policies)]

    # This function takes in a context and an action and return a VW format string that contains the given inputs
    def vw_format(self, context, action, reward=None): 
        vw_example = ""

        if reward is not None:
            vw_example += f"{reward} "

        # Add server load (l stands for load)
        vw_example += "|l"

        for server_index in range(self.num_actions):
            vw_example += f" load{server_index}:{context[server_index]}"

        # Add the chosen action
        vw_example += f" |a server{action}"

        return vw_example

    # Updates Q value
    def update_Q(self, policy_index, index, state, action, reward):
         self.models[policy_index][index].learn(self.vw_format(state, action, reward))

    # Retrieves a Q value from the VW model
    def get_Q(self, policy_index, index, state, action):
        if index > self.horizon - 1:
            return 0
        return(self.models[policy_index][index].predict(self.vw_format(state, action)))

    # Retrieves a V value from the VW model
    def get_V(self, policy_index, index, state, target_action_probabilities):
        if index > self.horizon - 1:
            return 0
        
        V = 0
        for action_index, action_prob in enumerate(target_action_probabilities):
            V += self.get_Q(policy_index, index, state, action_index) * action_prob
        return V

    """
    Implement IS, stepwise IS, WIS, stepwise WIS
    """
    def evaluate_one_trajectory(self, trajectory):
        self.num_trajectories += 1

        trajectory_per_step_cumulative_importance_ratios = [[] for _ in range(self.num_target_policies)]

        for policy_index in range(self.num_target_policies):
            policy = self.target_policies[policy_index][1]

            # terms of trajectory-wise DR estimates
            first_term = 0
            second_term = 0
            third_term = 0

            # sample[0] is an array of the server_load and (action, reward) revealed by each time step
            # sample[1] is the action taken at each time step
            # sample[2] is the reward of the action 
            # sample[3] is an array of action probabilities in each time step
            for index, sample in enumerate(trajectory):
                state = sample[0]
                action = sample[1]
                reward = sample[2]
                action_probabilities = sample[3]

                for sample_action, sample_reward in state[1]:
                    policy.update(sample_action, sample_reward)

                target_action_probabilities = policy.choose_action(context=state[0], return_prob=True, behavior_policy=False, chosen_action=action)
                importance_ratio = target_action_probabilities[action] / action_probabilities[action]               
                if (index == 0):
                    trajectory_per_step_cumulative_importance_ratios[policy_index].append(importance_ratio)
                else:
                    trajectory_per_step_cumulative_importance_ratios[policy_index].append(importance_ratio * trajectory_per_step_cumulative_importance_ratios[policy_index][-1])

                
                # First term for trajectory-wise DR estimate
                first_term += trajectory_per_step_cumulative_importance_ratios[policy_index][-1] * reward    
                
                # Second term
                second_term += trajectory_per_step_cumulative_importance_ratios[policy_index][-1] * self.get_Q(policy_index, index, state[0], action)
                
                # Third term
                if (index == 0):
                    third_term += self.get_V(policy_index, index, state[0], target_action_probabilities)
                else:
                    third_term += trajectory_per_step_cumulative_importance_ratios[policy_index][-2] *  self.get_V(policy_index, index, state[0], target_action_probabilities)
                    
            self.DR_estimates[policy_index].append(first_term - second_term + third_term)

            # Updates the VW model with the trajectory
            for index in range(len(trajectory)-1):
                state = trajectory[index][0]
                action = trajectory[index][1]
                reward = trajectory[index][2]
                next_state_V_estimate = self.get_V(policy_index, index + 1, trajectory[index+1][0][0], trajectory[index+1][3])

                self.update_Q(policy_index, index, state[0], action, reward + next_state_V_estimate) 
            self.update_Q(policy_index, index, trajectory[-1][0][0], trajectory[-1][1], trajectory[-1][2])   
            
            policy.reset()
        
        self.per_trajectory_cumulative_importance_ratios.append(trajectory_per_step_cumulative_importance_ratios)
        self.rewards.append([row[2] for row in trajectory])

        return 

    # Returns the mean IS estimate for however many trajectories were already analyzed 
    def get_IS_estimate(self, policy_index):
        # each IS estimator is a product of the cumulative importance ratios and the sum of the rewards
        sum_rewards_by_trajectory = np.sum(self.rewards, axis = 1) 
        product_rhos_by_trajectory = [row[policy_index][-1] for row in self.per_trajectory_cumulative_importance_ratios]
        IS_estimate_by_trajectory = np.multiply(product_rhos_by_trajectory,sum_rewards_by_trajectory)
        return np.mean(IS_estimate_by_trajectory)

    # Returns the mean step-IS estimate for however many trajectories were already analyzed 
    def get_stepIS_estimate(self, policy_index):
        # each step-IS estimator is a dot product of the cumulative importance ratios at each time step and the reward at each time step
        new_per_step_importance_ratios = [row[policy_index] for row in self.per_trajectory_cumulative_importance_ratios]
        stepIS_estimate_by_trajectory = [np.dot(self.rewards[t], new_per_step_importance_ratios[t]) for t in range(len(self.rewards))]
        return np.mean(stepIS_estimate_by_trajectory)

    # Returns the mean WIS estimate for however many trajectories were already analyzed 
    def get_WIS_estimate(self, policy_index):
        sum_rewards_by_trajectory = np.sum(self.rewards, axis = 1)
        product_rhos_by_trajectory = [row[policy_index][-1] for row in self.per_trajectory_cumulative_importance_ratios]
        WIS_estimate_by_trajectory = np.zeros(len(sum_rewards_by_trajectory))
        
        avg_cumulative_importance_ratios = np.mean([row[policy_index] for row in self.per_trajectory_cumulative_importance_ratios], axis=0)
        if (avg_cumulative_importance_ratios[-1]):
            WIS_estimate_by_trajectory = product_rhos_by_trajectory / avg_cumulative_importance_ratios[-1] * sum_rewards_by_trajectory

        return np.mean(WIS_estimate_by_trajectory)

    # Returns the mean step-WIS estimate for however many trajectories were already analyzed 
    def get_stepWIS_estimate(self, policy_index):
        new_per_step_importance_ratios = [row[policy_index] for row in self.per_trajectory_cumulative_importance_ratios]

        avg_cumulative_importance_ratios = np.mean([row[policy_index] for row in self.per_trajectory_cumulative_importance_ratios], axis=0)
        new_importance_weights = [np.divide(new_per_step_importance_ratios[t], avg_cumulative_importance_ratios, \
                                            out=np.zeros_like(new_per_step_importance_ratios[t]), where=avg_cumulative_importance_ratios!=0) 
                                            for t in range(len(self.rewards))]
        
        stepWIS_estimate_by_trajectory = [np.dot(self.rewards[t], new_importance_weights[t]) for t in range(len(self.rewards))]
        return np.mean(stepWIS_estimate_by_trajectory)

    def get_DR_estimate(self, policy_index):
        return np.mean(self.DR_estimates[policy_index])