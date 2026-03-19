import random

def greedy_policy(state, q):
    actions = list(q[state].keys())
    return max(actions, key=lambda action: q[state][action])

def epsilon_greedy_policy(state, q, epsilon):
    actions = list(q[state].keys())

    if random.random() < epsilon:
        return random.choice(actions)

    return greedy_policy(state, q)