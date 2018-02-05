import gym
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt


env = gym.make('FrozenLake8x8-v0')
tf.reset_default_graph
inputs1 = tf.placeholder(shape=[1, 64], dtype=tf.float32)
W = tf.Variable(tf.random_uniform([64, 4], 0, 0.01))
Qout = tf.matmul(inputs1, W)
predict = tf.argmax(Qout, 1)
nextQ = tf.placeholder(shape=[1, 4], dtype=tf.float32)
loss = tf.reduce_sum(tf.square(nextQ - Qout))
trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
updateModel = trainer.minimize(loss)
init = tf.global_variables_initializer()

y = 0.99
e = 0.1
num_episodes = 20000

jList = []
rList = []


with tf.Session() as sess:
    sess.run(init)
    for i in range(num_episodes):
        s = env.reset()
        rAll = 0
        d = False
        j = 0
        while j < 500:
            j += 1
            # 前回のstatusからQ値を予測
            a, allQ = sess.run([predict, Qout], feed_dict={
                               inputs1: np.identity(64)[s:s + 1]})
            # εグリーディー(ランダムな手を打たせる）
            if np.random.rand(1) < e:
                a[0] = env.action_space.sample()
            s1, r, d, _ = env.step(a[0])
            # 次のQ値の推定値を求める
            Q1 = sess.run(Qout, feed_dict={
                          inputs1: np.identity(64)[s1:s1 + 1]})
            maxQ1 = np.max(Q1)
            targetQ = allQ
            # Q値のターゲットを更新する
            targetQ[0, a[0]] = r + y * maxQ1

            # 新しい重みを求める
            _, W1 = sess.run([updateModel, W], feed_dict={
                             inputs1: np.identity(64)[s:s + 1], nextQ: targetQ})
            rAll += r
            s = s1
            if d == True:
                # イプシロン（exploreの閾値）を更新
                e = 1.0 / ((i / 500) + 10)
                break

        print(str(i) + "番目の試行が終わりました")
        jList.append(j)  # 試行回数のリスト
        rList.append(rAll)  # 報酬のリスト

print("Success Episode Ratio: " + str(sum(rList) / num_episodes * 100) + "%")

plt.plot(rList)
plt.plot(jList)
