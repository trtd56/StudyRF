# OpenAIGymで強化学習導入

## 事前準備

### minicondaのインストール

pythonプログラムが実行できればなんでもよいが、bash on windowsなどは画面を飛ばす設定などが面倒なので、Anacondaがお勧め。  
ここでは最小構成のminicondaをインストール。

[公式サイト](https://conda.io/miniconda.html)から自分の環境にあったものをインストールする。  
コードは3.6で書いているので、特にこだわりがなければ3系で。

### 必要なpythonモジュールのインストール

~~~
$ pip install numpy chainer gym chainerrl ipython
~~~

## サンプル

### OpenAIGym環境

#### 学習環境

pythonインタプリタかipythonを使ってCartPoleの動作を確認する。  
CartPoleは倒立振子で、OpenAIGymで最も簡単な環境のうちの一つ。

##### 環境の作成

CartPoleの環境を作成する。  
env.reset()はCartPoleの台座を初期位置に戻すが、ランダム性があり、毎回若干違う位置にリセットされる。

~~~python
$ python
>>>> import gym  # OpenAIGymをインポート
>>>> env = gym.make('CartPole-v0')  # CartPoleの環境を作成
>>>> observation = env.reset()  # 環境をリセット
>>>> print(observation)
[-0.04459665  0.0165073   0.04207109  0.01156669]
~~~
##### 環境の更新

~~~python
>>>> action = env.action_space.sample()  # 行動のサンプルを取得
>>>> print(action)
1
>>>> observation, reward, done, info = env.step(action)  # サンプリングした行動で環境を更新
>>>> print(observation)  更新後の環境
[-0.04426651  0.21100144  0.04230242 -0.26755111]
>>>> print(reward)  # 報酬
1
>>>> print(done)  # エピソードが終了したかどうか
False
>>>> print(info)  # 環境情報(今回は使わない)
{}
~~~
### エージェント

サンプルのエージェントを2個作った

#### random_agent

ランダムに動くエージェント。

#### dqn_agent

Deep Q-Networkを使ったエージェント。  
DoubleDQNを使っている。
