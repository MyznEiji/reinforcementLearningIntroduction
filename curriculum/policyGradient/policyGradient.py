""" 多腕バンディット問題"""

import tensorflow as tf
import numpy as np

# 値が小さいほど報酬を得やすい
bandits = [0.2, -0.5, -0.2, 0]
num_bandits = len(bandits)

# アームを引いて報酬を返す関数を定義
# 乱数の値がbandits(閾値)より大きいと1を返す

def pullBandit(bandit):
    result = np.random.randn(1)
    if result > bandit:
        return 1
    else:
        return -1

# グラフを定義する


tf.reset_default_graph()

# 各アームごとの重みを格納する
weights = tf.Variable(tf.ones([num_bandits]))
# 今回引くアームの番号を入れる
chosen_action = tf.argmax(weights, 0)
# リワードを格納する
reward_holder = tf.placeholder(shape=[1], dtype=tf.float32)
# アクションを保持する変数（整数）
action_holder = tf.placeholder(shape=[1], dtype=tf.int32)
# 選択したアクションの重み取り出す
responsible_weight = tf.slice(weights, action_holder, [1])
# 損失関数
loss = -(tf.log(responsible_weight) * reward_holder)
# learning_rate の値を小さくしすぎると学習が進まない。 大きくすると値が発散しやすくなる
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
# targetに近くなるように損失関数を最適化
update = optimizer.minimize(loss)


# エージェントをトレーニングする

total_episodes = 1000
total_reward = np.zeros(num_bandits)
e = 0.1
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    i = 0
    while i < total_episodes:
        if np.random.rand(1) < e:
            action = np.random.randint(num_bandits)
        else:
            action = sess.run(chosen_action)
        reward = pullBandit(bandits[action])

        _, resp, ww = sess.run([update, responsible_weight, weights], feed_dict={
                               reward_holder: [reward], action_holder: [action]})

        total_reward[action] += reward
        if i % 50 == 0:
            print("リワード・報酬の一覧：" + str(total_reward))
        i += 1

print("エージェントが考える最適なアームは、" + str(np.argmax(ww) + 1) + "番目のアームです。")
