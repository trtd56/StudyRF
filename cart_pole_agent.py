# -*- cording: utf-8 -*-

import gym
import sys

env = gym.make('CartPole-v0')

print('episode\tn_step\treward')
for i in range(5):  # 5エピソード試行を行う
    obs = env.reset()
    done = False
    R = 0  # 合計報酬
    t = 0  # 継続したステップ数
    while not done and t < 200:  # 最大200ステップで終了
        env.render()
        if obs[3] < 0:
            action = 0
        else:
            action = 1
        obs, r, done, _ = env.step(action)
        R += r  # 報酬を加算していく
        t += 1
    print('{0}\t{1}\t{2}'.format(i, t, R))
