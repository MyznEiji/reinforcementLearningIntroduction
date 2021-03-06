import gym
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt


env = gym.make('FrozenLake-v0')

# tensorflowの初期化
tf.reset_default_graph
# 入力値の格納するデータ
inputs1 = tf.placeholder(shape=[1, 16], dtype=tf.float32)
# 重み input1をQ値に変換する
W = tf.Variable(tf.random_uniform([16, 4], 0, 0.01))
# Q値の計算値を格納する
Qout = tf.matmul(inputs1, W)
# 報酬値の推定 ,Qoutを最大値にする予測値
predict = tf.argmax(Qout, 1)
# Q値推定値
nextQ = tf.placeholder(shape=[1, 4], dtype=tf.float32)
# 損失関数
loss = tf.reduce_sum(tf.square(nextQ - Qout))
# 勾配降下法のモデルの定義
trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
# lossを最小化するように計算する
updateModel = trainer.minimize(loss)
# 変数の初期化
init = tf.global_variables_initializer()

# 割引率
y = 0.99
# 選択するアクション値をきめるためのパラメータ
e = 0.1
num_episodes = 3000

# ステップごとの報酬を格納する
jList = []
# トータルの報酬を格納する
rList = []

with tf.Session() as sess:
    sess.run(init)
    for i in range(num_episodes):
        s = env.reset()
        rAll = 0
        d = False
        j = 0
        while j < 99:
            j += 1
            # 前回のstatusからQ値を予測
            a, allQ = sess.run([predict, Qout], feed_dict={
                               inputs1: np.identity(16)[s:s + 1]})
            # εグリーディー(ランダムな手を打たせる）
            if np.random.rand(1) < e:
                a[0] = env.action_space.sample()
            s1, r, d, _ = env.step(a[0])
            # 次のQ値の推定値を求める
            Q1 = sess.run(Qout, feed_dict={
                          inputs1: np.identity(16)[s1:s1 + 1]})
            maxQ1 = np.max(Q1)
            targetQ = allQ
            # Q値のターゲットを更新する
            targetQ[0, a[0]] = r + y * maxQ1

            # 新しい重みを求める
            _, W1 = sess.run([updateModel, W], feed_dict={
                             inputs1: np.identity(16)[s:s + 1], nextQ: targetQ})
            rAll += r
            s = s1
            if d == True:
                # イプシロン（exploreの閾値）を更新
                e = 1.0 / ((i / 50) + 10)
                break
        jList.append(j)  # 試行回数のリスト
        rList.append(rAll)  # 報酬のリスト

print("Success Episode Ratio: " + str(sum(rList) / num_episodes * 100) + "%")

plt.plot(rList)
plt.plot(jList)
