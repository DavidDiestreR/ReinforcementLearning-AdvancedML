import random

def policy(state, q, epsilon):
    actions = list(q[state].keys())

    if random.random() < epsilon:
        return random.choice(actions)

    return max(actions, key=lambda action: q[state][action])