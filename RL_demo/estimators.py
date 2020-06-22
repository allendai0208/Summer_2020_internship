import numpy as np

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
        self.per_step_importance_ratios = [] # Holds the per-step importance ratios of each action for each policy for each trajectory
        self.avg_cumulative_importance_ratios = [np.zeros(self.horizon) for _ in range(self.num_target_policies)]

    """
    Implement IS, stepwise IS, WIS, stepwise WIS
    """
    def evaluate_one_trajectory(self, trajectory):
        self.num_trajectories += 1

        trajectory_per_step_importance_ratios = [[] for _ in range(self.num_target_policies)]

        for policy_index in range(self.num_target_policies):
            policy = self.target_policies[policy_index][1]
            num_action = 1

            # sample[0] is an array of the server_load and (action, reward) revealed by each time step
            # sample[1] is the action taken at each time step
            # sample[2] is the reward at each time step
            # sample[3] is an array of action probabilities in each time step
            for sample in trajectory:
                action = sample[1]
                reward = sample[2]

                importance_ratio = policy.choose_action(context=sample[0][0], return_prob=True, behavior_policy=False, chosen_action=action)[action] / sample[3][action]               
                trajectory_per_step_importance_ratios[policy_index].append(importance_ratio)
            
                self.avg_cumulative_importance_ratios[policy_index][num_action - 1] = \
                    (self.avg_cumulative_importance_ratios[policy_index][num_action-1] * (self.num_trajectories - 1) + np.prod(trajectory_per_step_importance_ratios[policy_index])) / self.num_trajectories
                
                num_action += 1
                policy.update(action, reward) 

            policy.reset()
        
        self.per_step_importance_ratios.append(trajectory_per_step_importance_ratios)
        self.rewards.append([row[2] for row in trajectory])

        return 

    def get_rolling_product(self, A):
        prod = 1
        B = []
        for elem in A:
            prod *= elem
            B.append(prod)
        return B

    def get_IS_estimates(self, policy_index):
        sum_rewards_by_trajectory = np.sum(self.rewards, axis = 1) 
        product_rhos_by_trajectory = np.prod([row[policy_index] for row in self.per_step_importance_ratios], axis=1)
        IS_estimate_by_trajectory = np.multiply(product_rhos_by_trajectory,sum_rewards_by_trajectory)
        #print('sum_rewards_by_trajectory:',sum_rewards_by_trajectory)
        #print('product_rhos_by_trajectory:',product_rhos_by_trajectory)
        #print('IS_estimate_by_trajectory(', policy_index,'):',np.mean(IS_estimate_by_trajectory))
        return np.mean(IS_estimate_by_trajectory)


    def get_stepIS_estimates(self, policy_index):
        new_per_step_importance_ratios = [self.get_rolling_product(row[policy_index]) for row in self.per_step_importance_ratios]
        stepIS_estimate_by_trajectory = [np.dot(self.rewards[t], new_per_step_importance_ratios[t]) for t in range(len(self.rewards))]
        return np.mean(stepIS_estimate_by_trajectory)

    
    def get_WIS_estimates(self, policy_index):
        sum_rewards_by_trajectory = np.sum(self.rewards, axis = 1)
        product_rhos_by_trajectory = np.prod([row[policy_index] for row in self.per_step_importance_ratios], axis=1)
        WIS_estimate_by_trajectory = np.zeros(len(sum_rewards_by_trajectory))
        if (self.avg_cumulative_importance_ratios[policy_index][-1]):
            WIS_estimate_by_trajectory = product_rhos_by_trajectory / self.avg_cumulative_importance_ratios[policy_index][-1] * sum_rewards_by_trajectory
        return np.mean(WIS_estimate_by_trajectory)


    def get_stepWIS_estimates(self, policy_index):
        new_per_step_importance_ratios = [self.get_rolling_product(row[policy_index]) for row in self.per_step_importance_ratios]
        new_importance_weights = [np.divide(new_per_step_importance_ratios[t], self.avg_cumulative_importance_ratios[policy_index], \
                                            out=np.zeros_like(new_per_step_importance_ratios[t]), where=self.avg_cumulative_importance_ratios[policy_index]!=0) 
                                            for t in range(len(self.rewards))]
        stepWIS_estimate_by_trajectory = [np.dot(self.rewards[t], new_importance_weights[t]) for t in range(len(self.rewards))]
        return np.mean(stepWIS_estimate_by_trajectory)