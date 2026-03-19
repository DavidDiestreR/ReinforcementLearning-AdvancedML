import random


def _best_actions(state, q):
    actions = list(q[state].keys())
    best_value = max(q[state][action] for action in actions)
    return [action for action in actions if q[state][action] == best_value]


def greedy_policy(state, q):
    return _best_actions(state, q)[0]


def epsilon_greedy_policy(state, q, epsilon):
    actions = list(q[state].keys())

    if random.random() < epsilon:
        return random.choice(actions)

    return greedy_policy(state, q)
