from simulation import run_simulation
import numpy as np
from policies import *
from environment import Servers
from estimators import Evaluation
import sys
import matplotlib.pyplot as plt
import math
from matplotlib.transforms import ScaledTranslation

if __name__ == "__main__":
    # Initialize basic variables
    num_seeds = 5     #int(input("num_seeds: "))
    num_requests = 5 #int(input("num_requests: "))
    num_servers = 3   #int(input("num_servers: "))
    threshold = int(sys.argv[1])#int(input("threshold: "))
    #num_trajectories_list = [0] + [10 ** i for i in range(1, 7)]
    num_trajectories_list = [0] + [10 ** i for i in range(1, 6)]
    start_trajectory_index = 0
    # end_trajectory_index = 1
    #noise_type = 'increasing_variance'

    # Initialize policies
    '''
    behavior_policy = ('UniRan', UniRan(num_servers))
    target_policy1 = ('LeastLoad_0.3', LeastLoad(num_servers, epsilon=0.3))
    target_policy2 = ('LeastLoad_0', LeastLoad(num_servers, epsilon=0))
    target_policy3 = ('EpsilonGreedy', EpsilonGreedy(num_servers, epsilon=0.3))
    target_policy4 = ('UCB1', UCB1(num_servers))
    '''

    behavior_policy = ('EpsilonGreedy', EpsilonGreedy(num_servers, epsilon=0.3))
    target_policy1 = ('LeastLoad_0.3', LeastLoad(num_servers, epsilon=0.3))
    target_policy2 = ('LeastLoad_0', LeastLoad(num_servers, epsilon=0))
    target_policy3 = ('UniRan', UniRan(num_servers))
    target_policy4 = ('UCB1', UCB1(num_servers))


    target_policies = [target_policy1, target_policy2, target_policy3, target_policy4]
    num_target_policies = len(target_policies)

    policies = [target_policy1, target_policy2, target_policy3, target_policy4, behavior_policy]
    num_policies = 1 + num_target_policies

    target_policy_names = [policy[0] for policy in policies if policy != policies[-1]]

    # Initialize environment (servers)
    servers = Servers(num_policies=num_policies, num_servers=num_servers)

    # Initialize evaluation for each seed
    evaluations = [Evaluation(horizon=num_requests,
                              num_target_policies=num_target_policies,
                              num_actions=num_servers,
                              target_policies=target_policies)
                   for _ in range(num_seeds)]

    # Write output to a file
    file_name = f"num_requests_{num_requests}_num_servers_{num_servers}_threshold_{threshold}_no_noise_change.txt" #input("Enter file name: ")
    with open(file_name, 'w') as wf:
        
        IS_errors = [[[] for _ in range(len(num_trajectories_list) - 1)] for _ in range(num_seeds)]   
        stepIS_errors = [[[] for _ in range(len(num_trajectories_list) - 1)] for _ in range(num_seeds)]   
        WIS_errors = [[[] for _ in range(len(num_trajectories_list) - 1)] for _ in range(num_seeds)]   
        stepWIS_errors = [[[] for _ in range(len(num_trajectories_list) - 1)] for _ in range(num_seeds)]   

        cumulative_latency_per_policy = [np.zeros(num_target_policies) for _ in range(num_seeds)]

        # Iterate over number of trajectories
        for num_trajectories_index in range(1, len(num_trajectories_list)):
            write_string = f"\nnum_seeds: {num_seeds}" + f"\nnum_trajectory: {num_trajectories_list[num_trajectories_index]}" \
                           + f"\nnum_requests: {num_requests}" \
                           + f"\nnum_servers: {num_servers}" + f"\nbehavior policy: {behavior_policy[0]}" \
                           + f"\ntarget policies: {target_policy_names}"
            print(write_string)
            wf.write(write_string)


            # Iterate over random seed
            seed_add = 0
            for seed in range(seed_add, seed_add + num_seeds):
                # Set a numpy random seed
                np.random.seed(seed)

                # Set a particular evaluation for the seed
                seed_evaluation = evaluations[seed - seed_add]

                # Store the estimates to calculate their errors
                IS_estimates_per_seed = np.zeros(num_target_policies)
                stepIS_estimates_per_seed = np.zeros(num_target_policies)
                WIS_estimates_per_seed = np.zeros(num_target_policies)
                stepWIS_estimates_per_seed = np.zeros(num_target_policies)

                # Store the errors to plot later
                IS_errors_per_seed = np.zeros(num_target_policies)
                stepIS_errors_per_seed = np.zeros(num_target_policies)
                WIS_errors_per_seed = np.zeros(num_target_policies)
                stepWIS_errors_per_seed = np.zeros(num_target_policies)

                # Iterate over trajectories of the given number to run simulation
                for trajectory_index in range(num_trajectories_list[num_trajectories_index - 1], num_trajectories_list[num_trajectories_index]):
                    # print(f"seed:{seed - seed_add} / trajectory: {seed_evaluation.num_trajectories}", end='\r')

                    # Run simulation and get trace & true performances of every target policy
                    total_latency_for_each_policy, trace = \
                        run_simulation(policies=policies, num_requests=num_requests, servers=servers)

                    # Get the estimate of every target policy on the trajectory. 
                    """
                    Implement
                    """
                    cumulative_latency_per_policy[seed-seed_add] += total_latency_for_each_policy[:-1]
                    seed_evaluation.evaluate_one_trajectory(trace)
                    

                for policy_index in range(num_target_policies):
                    IS_estimates_per_seed[policy_index] = seed_evaluation.get_IS_estimates(policy_index)
                    stepIS_estimates_per_seed[policy_index] = seed_evaluation.get_stepIS_estimates(policy_index)
                    WIS_estimates_per_seed[policy_index] = seed_evaluation.get_WIS_estimates(policy_index)
                    stepWIS_estimates_per_seed[policy_index] = seed_evaluation.get_stepWIS_estimates(policy_index)

                    true_performance = cumulative_latency_per_policy[seed-seed_add][policy_index]/num_trajectories_list[num_trajectories_index]
                    #print(true_performance)

                    IS_errors_per_seed[policy_index] = (true_performance - IS_estimates_per_seed[policy_index]) / true_performance
                    stepIS_errors_per_seed[policy_index] = (true_performance - stepIS_estimates_per_seed[policy_index]) / true_performance
                    WIS_errors_per_seed[policy_index] = (true_performance - WIS_estimates_per_seed[policy_index]) / true_performance
                    stepWIS_errors_per_seed[policy_index] = (true_performance - stepWIS_estimates_per_seed[policy_index]) / true_performance

                IS_errors[seed - seed_add][num_trajectories_index - 1] = IS_errors_per_seed
                stepIS_errors[seed - seed_add][num_trajectories_index - 1] = stepIS_errors_per_seed
                WIS_errors[seed - seed_add][num_trajectories_index - 1] = WIS_errors_per_seed
                stepWIS_errors[seed - seed_add][num_trajectories_index - 1] = stepWIS_errors_per_seed

        # After finishing iterating over every random seed,
        # display the estimate of each target policy for the given number of trajectories
        
        IS_mean_errors = np.mean(IS_errors, axis=0)
        stepIS_mean_errors = np.mean(stepIS_errors, axis=0)
        WIS_mean_errors = np.mean(WIS_errors, axis=0)
        stepWIS_mean_errors = np.mean(stepWIS_errors, axis=0)
        
        IS_std = np.std(IS_errors, axis=0)
        stepIS_std = np.std(stepIS_errors, axis=0)
        WIS_std = np.std(WIS_errors, axis=0)
        stepWIS_std = np.std(stepWIS_errors, axis=0)

        x = [math.pow(10, i) for i in range(1, 6)]

        for policy_index in range(num_target_policies):
            fig, ax = plt.subplots()
            trans1 = ax.transData + ScaledTranslation(-6/72, 0, fig.dpi_scale_trans)
            trans2 = ax.transData + ScaledTranslation(-2/72, 0, fig.dpi_scale_trans)
            trans3 = ax.transData + ScaledTranslation(+2/72, 0, fig.dpi_scale_trans)
            trans4 = ax.transData + ScaledTranslation(+6/72, 0, fig.dpi_scale_trans)

            ax.errorbar(x, [row[policy_index] for row in IS_mean_errors], yerr=[row[policy_index] for row in IS_std], transform=trans1, capsize=3, capthick=1, linewidth=1, marker='x', label='IS')

            ax.errorbar(x, [row[policy_index] for row in stepIS_mean_errors], yerr=[row[policy_index] for row in stepIS_std], transform=trans2, capsize=3, capthick=1, linewidth=1, marker='x', label='stepIS')
                             
            ax.errorbar(x, [row[policy_index] for row in WIS_mean_errors], yerr=[row[policy_index] for row in WIS_std], transform=trans3, capsize=3, capthick=1, linewidth=1, marker='x', label='WIS')
            
            ax.errorbar(x, [row[policy_index] for row in stepWIS_mean_errors], yerr=[row[policy_index] for row in stepWIS_std], transform=trans4, capsize=3, capthick=1, linewidth=1, marker='x', label='stepWIS')
            

            plt.ylabel('Mean errors of estimators')
            plt.xlabel('Number of samples (log10)')
            plt.xscale('log')
            plt.legend()

            plt.show()



                
            
