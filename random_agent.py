# -*- cording: utf-8 -*-

import gym
env = gym.make('CartPole-v0')

for i_episode in range(20):  # 合計20エピソード環境を実行する
    observation = env.reset()
    R = 0  # エピソード毎の報酬の合計値
    for t in range(100):  # 1エピソード最大100ステップ環境を更新する
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        R += reward  # エピソードの報酬を足していく
        if done:  # CartPoleの棒の傾きが一定以上になるとエピソード終了
            print("Episode {} finished. sum of reward is {}".format(i_episode, R))
            break
