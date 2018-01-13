# -*- cording: utf-8 -*-

import gym
import time

env = gym.make('CartPole-v0')
obs = env.reset()

done = False
print('X\tVx\tΘ\tVΘ\treward\taction')
for i in range(200):
    env.render()
    action = env.action_space.sample()
    obs, r, done, _ = env.step(action)
    log = '{2:01.3f}\t{3:01.3f}\t{4:01.3f}\t{5:01.3f}\t{0}\t{1}'.format(r, action, *obs)
    print(log)
    if done:
        break
    time.sleep(0.5)
