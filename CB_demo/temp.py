import numpy as np 
from matplotlib import pyplot as plt

np.random.seed(0)

# Initialize an environment

# Each user is a context
# Each user is denoted by his/her index (ex: user 0 or user 3)
num_users = 5 

# Each advertisment is an action
num_ads = 10 # There are 10 ads 

# reward_table is a dictionary of key: the index of a user / value: np.array of the deterministic reward of each action (ad)
reward_table = {user: np.random.choice(10, size=num_ads) for user in range(num_users)}

# Implement IPS 
class IPS:
    def __init__(self, trace):
        self.trace = trace
    
    def get_estimate(self):
        latest = {}
        for sample in reversed(self.trace):
            if (sample[0], sample[1]) not in latest:
                latest[(sample[0],sample[1])] = (sample[2], sample[3],sample[4])
        n = len(latest)
        total = 0
        for valid_IPS_sample in latest:
            weight = latest[valid_IPS_sample][1]/latest[valid_IPS_sample][0]
            reward = latest[valid_IPS_sample][2]
            total += weight*reward
        return total / n
                

# Three policies    
class UniformlyRandom:
    def __init__(self, num_actions):
        """
        num_actions: number of actions (in this demo, it is equal to the number of ads)
        """
        self.num_actions = num_actions 

    def choose_action(self, context ):
        """
        return a chosen action and action-probabilities
        """
        return np.random.choice(self.num_actions), [1 / self.num_actions] * self.num_actions

    def update(self, cb_sample):
        """
        UniformlyRandom is a stateless policy, which means it does not update itself
        based on the past rewards revelaed by it.

        (parameters)
        cb_sample: a tuple of (context, chosen action, reward revealed)
        """
        pass

    def toString(self):
        return 'UniformlyRandom'

class EpsilonGreedy:
    def __init__(self, num_actions, num_contexts, epsilon=0.2):
        """
        This is a CB verision of epsilon greedy

        (parameters)
        num_actions: number of actions (= number of ads)
        num_contexts: number of contexts (= number of users)
        epsilon: this policy chooeses a random action with epsilon probability 

        (Stored values)
        self.num_actions = num_actions 
        self.num_contexts = num_contexts
        self.num_actions_chosen: keep track of how many times each action is chosen for each context
        self.sum_rewards: keep track of the total reward of each action for each context

        The reason why self.num_actions_chosen is initialized with np.ones instead of np.zeros is
        to prevent division by zero in the function choose_action (line 68) 
        when an action has never been chosen so far for a given context.
        """
        self.num_actions = num_actions 
        self.num_contexts = num_contexts
        self.num_actions_chosen = {context: np.ones(num_actions) for context in range(num_contexts)}
        self.sum_rewards = {context: np.zeros(num_actions) for context in range(num_contexts)}
        self.epsilon = epsilon

    def choose_action(self, context):
        """
        return a chosen action and its probability
        """
        # Check which action is the best action 
        # The best action is the action that has the highest total reward for a given context
        best_action = np.argmax(self.sum_rewards[context] / self.num_actions_chosen[context])
        #best_action = np.argmax(self.sum_rewards)

        # Get action probabilities based on the best action
        # there is a (epsilon + epsilon/num_actions) probability of choosing the best action.
        # there is a (epsilon/num_actions) probability of choosing any other action
        action_probabilities = [self.epsilon / self.num_actions] * self.num_actions
        action_probabilities[best_action] += 1 - self.epsilon

        # Sample an action
        chosen_action = np.random.choice(self.num_actions, p=action_probabilities)

        # Record the number of times the chosen action is chosen
        self.num_actions_chosen[context][chosen_action] += 1

        return chosen_action, action_probabilities

    def update(self, cb_sample):
        """
        EpsilonGreedy is a stateful policy, which means it does update itself
        based on the past rewards revelaed by it.
        Thus, update function is necessary.

        (parameters)
        cb_sample: a tuple of (context, chosen action, reward revealed)
        """
        context, chosen_action, reward_revealed = cb_sample

        self.sum_rewards[context][chosen_action] += reward_revealed

    def toString(self):
        return 'EpsilonGreedy'

# Implement your own simple stochastic policy (This should be different form the uniformly random policy)
class StochasticPolicy:
    def __init__(self):
        pass

    def choose_action(self, context):
        pass

    def update(self, cb_sample):
        pass

# Implement your own simple deterministic policy
class DeterministicPolicy:
    def __init__(self, num_actions):
        self.num_actions = num_actions

    def choose_action(self, context):
        action_probabilities = np.zeros(self.num_actions)
        action_probabilities[0] = 1
        return 0, action_probabilities

    def update(self, cb_sample):
        pass

# Initialize policies
epsilon = 0.3 # epsilon is a tunable parameter

uni_ran = UniformlyRandom(num_ads)
eps_greedy = EpsilonGreedy(num_ads, num_users, epsilon=epsilon)

# Start the demo
num_samples = 1000000 # total number of samples. Feel free to change this if you want to
trace = list() # list to store each CB tuple of (context, chosen action, action probability, reward revealed)
new_policy_true_rewards = list() # Keep track of true rewards revealed by the new policy

# Implement the demo as specified by the comments below
old_policy = uni_ran
new_policy = eps_greedy

for sample_index in range(num_samples): 
    # Context (user) revealed
    user = np.random.choice(num_users)

    # The old policy chooses an action
    old_chosen_action, old_probabilities = old_policy.choose_action(user)

    # The reward of the action chosen by the old policy is revealed
    reward = reward_table[user][old_chosen_action]

    # The new policy chooses an action
    new_chosen_action, new_probabilities = new_policy.choose_action(user)

    # Record the reward revealed by the new policy to get the true performance of the new policy
    new_policy_true_rewards.append(new_chosen_action)

    # Store the CB sample in trace for evaluation
    trace.append((user, 
                  old_chosen_action, 
                  old_probabilities[old_chosen_action], 
                  new_probabilities[new_chosen_action], 
                  reward))

    # Update the old policy if needed
    old_policy.update((user, old_chosen_action, reward))

# The expected value of the reward revealed by the new policy
# Actually, this is not the "true" expected value but a value 
# calculated by monte carlo methods.
# If you really want to be accurate, you can directly calculate 
# the true expected value using the environment defined at the beginning.
true_performance = np.mean(new_policy_true_rewards)

# Implement a code to get IPS estimates using the trace
ips = IPS(trace)
estimate = ips.get_estimate()
print('IPS(' + old_policy.toString() + ', ' + new_policy.toString() + ') = ' + str(estimate))

# results:
#   IPS(UniformlyRandom, EpsilonGreedy) = 22.599999999999994
#   IPS(UniformlyRandom, StochasticPolicy)
#   IPS(UniformlyRandom, DeterministicPolicy)