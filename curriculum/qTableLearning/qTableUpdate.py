import gym
import numpy as np


env = gym.make('FrozenLake-v0')
Q = np.zeros([env.observation_space.n, env.action_space.n])

# パラメータを更新する値
lr = 0.8
# ハイパーパラメータ
y = 0.95
# 思考する回数
num_episodes = 2000
# 報酬のリスト
rList = []

for i in range(num_episodes):
    # ステートを初期化
    s = env.reset()
    # 報酬の合計
    rAll = 0
    d = False
    # エピソードあたりの移動数
    j = 0
    while j < 99:
        j += 1
        a = np.argmax(Q[s, :] + np.random.randn(
            1, env.action_space.n) * (1.0 / (i + 1)))
        s1, r, d, _ = env.step(a)
        Q[s, a] = Q[s, a] + lr * (r + y * np.max(Q[s1, :]) - Q[s, a])
        rAll += r
        s = s1
        if d == True:
            break
    rList.append(rAll)

print("回数ごとの結果:" + str(sum(rList)/num_episodes))
print("最終的なQテーブルの値", "\n", Q)
