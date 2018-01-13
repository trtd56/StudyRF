# -*- cording: utf-8 -*-

import gym
import sys

# 環境名の取得
args = sys.argv
env_name = args[1]
env = gym.make(env_name)

print('episode\tn_step\treward')
for i in range(5):
    obs = env.reset()
    done = False
    R = 0  # 合計報酬
    t = 0  # 継続したステップ数
    while not done and t < 200:  # 最大200ステップで終了
        env.render()
        action = env.action_space.sample()
        obs, r, done, _ = env.step(action)
        R += r  # 報酬を加算していく
        t += 1
    print('{0}\t{1}\t{2}'.format(i, t, R))
