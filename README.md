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

環境の各要素は下記の状態と対応している
![CartPole_state](https://github.com/trtd56/StudyRF/blob/master/pict/CartPole_state.png)

##### 環境の更新

~~~python
>>>> action = env.action_space.sample()  # 行動のサンプルを取得(0:カートを左へ移動/1:カートを右へ移動)
>>>> print(action)
1
>>>> observation, reward, done, info = env.step(action)  # サンプリングした行動で環境を更新
>>>> print(observation)  更新後の環境
[-0.04426651  0.21100144  0.04230242 -0.26755111]
>>>> print(reward)  # 報酬(Θが±0.209以内なら1)
1
>>>> print(done)  # エピソードが終了したかどうか(Θが±0.209を超えるとエピソード終了)
False
>>>> print(info)  # 環境情報(今回は使わない)
{}
~~~

##### 環境のサンプル実行

CartPoleの環境のサンプル実行プログラムを作った。  
実際の実行は一瞬で終わってしまうので、1描画毎に0.5秒のsleepを入れている。  
下記のように実行される。

~~~
(C:\Users\trtd\Miniconda3) C:\Users\trtd\StudyRF>python env_test.py
X       Vx      Θ      VΘ     reward  action
-0.018  -0.183  0.043   0.323   1.0     0
-0.022  0.011   0.050   0.044   1.0     1
-0.022  0.205   0.050   -0.233  1.0     1
　・
　・
　・
~~~

### エージェント

サンプルのエージェントを3個作った

#### random_agent

ランダムに動くエージェント。  
第1引数にClassic controlの環境名を指定する。

##### 指定できる環境
- CartPole-v0
- Acrobot-v1
- MountainCar-v0
- Pendulum-v0

##### 実行例(CartPole-v0の場合)

~~~bash
(C:\Users\trtd\Miniconda3) C:\Users\trtd\StudyRF>python random_agent.py CartPole-v0
episode n_step  reward
0       16      16.0
1       13      13.0
2       26      26.0
3       21      21.0
4       17      17.0
~~~

- episode: エピソード番号
- n_step: 継続したステップ数
- reward: 取得した合計報酬

#### cart_pole_agent

CartPoleに特化してハードコーディングしたエージェント。  
VΘが正ならば右、負か0ならば左に動く。

##### 実行例

~~~bash
(C:\Users\trtd\Miniconda3) C:\Users\trtd\StudyRF>python cart_pole_agent.py
episode n_step  reward
0       200     200.0
1       200     200.0
2       199     199.0
3       200     200.0
4       141     141.0
~~~

ほぼほぼ200ステップ継続させることができる。  
こういったことができるのは、CartPoleの世界を構成する要素をすべて把握することができるから。

#### dqn_agent

Deep Q-Networkを使ったエージェント。  
DoubleDQNを使っている。

## 作成したエージェントのアップロード

以前は作成したエージェントの結果を環境にアップロードして、報酬を競わせていたが、現在はあまり活動していないようである(リンク切れになってる部分が多い)

上位互換の[OpenAIUniverse](https://blog.openai.com/universe/)が登場したからだと思われる

## 参考

その他の環境を試したい方は[OpenAIGymの公式サイト](https://gym.openai.com/envs/)を参考にするとよい。

Atariなどは非常に面白そうだが、CNNの知識がある程度必要であり、GPUがない場合の学習時間が現実的ではないので、導入にはClassic controlがおすすめ。
