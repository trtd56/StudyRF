# -*- cording: utf-8 -*-

import chainer
import chainer.functions as F
import chainer.links as L
import chainerrl
import gym
import numpy as np

# 環境設定
env_name = 'CartPole-v0'
n_episodes = 200
max_episode_len = 200

# Agentパラメーター
n_hidden_channels = 50  # 隠れ層のユニット数
gamma = 0.95  # 報酬の割引率
epsilon = 0.3  # 探索と活用の割合(Epsilon-Greedy)
capacity = 10 ** 6 # 過去の経験をどれだけ覚えておくか(Experience Replay)
replay_start_size = 500  # どれだけ環境情報を得たら学習を始めるか
update_interval = 1  # ネットワークの更新頻度
target_update_interval = 100  # targetネットワークの更新(コピー)頻度

class QFunction(chainer.Chain):
    """
    Q関数
    通常のchainerでニューラルネットワークを定義するのと同じ
    """

    def __init__(self, obs_size, n_actions, n_hidden_channels):
        super().__init__()
        with self.init_scope():
            self.l0 = L.Linear(obs_size, n_hidden_channels)
            self.l1 = L.Linear(n_hidden_channels, n_hidden_channels)
            self.l2 = L.Linear(n_hidden_channels, n_actions)

    def __call__(self, x, test=False):
        h = F.tanh(self.l0(x))
        h = F.tanh(self.l1(h))
        return chainerrl.action_value.DiscreteActionValue(self.l2(h))

env = gym.make(env_name)  # 環境作成
obs_size = env.observation_space.shape[0]  # 環境の状態を示す変数の数を取得
n_actions = env.action_space.n  # とれる行動の数を取得
q_func = QFunction(obs_size, n_actions, n_hidden_channels)  # Q関数の定義

# 最適化手法の定義
# 通常のchainerと同じ
optimizer = chainer.optimizers.Adam()
optimizer.setup(q_func)

# 探索ルールを定義
# 今回はEpsilon-Greedy
explorer = chainerrl.explorers.ConstantEpsilonGreedy(
    epsilon, random_action_func=env.action_space.sample)
# Experience Replayのサイズを定義
replay_buffer = chainerrl.replay_buffer.ReplayBuffer(capacity)
# Agentに入ってくる環境情報をchainerが使えるnumpy.float32に変換する
phi = lambda x: x.astype(np.float32, copy=False) 
# Agent定義
# せっかくなのでDoubleDQNを使ってみる
agent = chainerrl.agents.DoubleDQN(
    q_func, optimizer, replay_buffer, gamma, explorer,
    replay_start_size=replay_start_size, update_interval=update_interval,
    target_update_interval=target_update_interval, phi=phi)

# 学習
# ここで毎回画面描画すると時間がかかるので10エピソード毎の結果を表示させる
print('start train')
for i in range(n_episodes):
    obs = env.reset()
    reward = 0
    done = False
    R = 0  # 報酬の合計値
    t = 0  # time step
    while not done and t < max_episode_len:
        action = agent.act_and_train(obs, reward)
        obs, reward, done, _ = env.step(action)
        R += reward
        t += 1
    if i % 10 == 0:
        print("train pisode {} finished. sum of reward is {}. agent statistics {}".format(
                                  i, R, agent.get_statistics()))
    agent.stop_episode_and_train(obs, reward, done)
print('train finished')

# テスト
# 探索は行わず、学習結果を使って行動を選択する
print('start test')
for i in range(10):
    obs = env.reset()
    done = False
    R = 0
    t = 0
    while not done and t < 200:
        env.render()
        action = agent.act(obs)
        obs, r, done, _ = env.step(action)
        R += r
        t += 1
    print('test episode:', i, 'R:', R)
    agent.stop_episode()
print('test finished')
