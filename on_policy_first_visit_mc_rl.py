#from src.environment import ...
#from src.policy import ...

# Q = on_policy_first_visit_mc_control(num_episodes, gamma, epsilon)
# Dentro del algoritmo, cada iteración que cae en un estado que no ocurre antes actualiza la siguiente Q para dicho estado
'''
Q[(x, y, vx, vy)] = {
    (-1, -1): ...,
    (-1, 0): ...,
    (-1, 1): ...,
    (0, -1): ...,
    (0, 0): ...,
    (0, 1): ...,
    (1, -1): ...,
    (1, 0): ...,
    (1, 1): ...
}
'''