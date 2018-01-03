# -*- cording: utf-8 -*-

import gym

env = gym.make('CartPole-v0')
#env = gym.make('MountainCar-v0')

for i in range(10):
    obs = env.reset()
    done = False
    R = 0
    t = 0
    while not done and t < 200:
        env.render()
        action = env.action_space.sample()
        obs, r, done, _ = env.step(action)
        R += r
        t += 1
    print('episode:', i, 'R:', R)
