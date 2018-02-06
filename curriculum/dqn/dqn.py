"""# カートポール問題を多層ニューラルネットワークで解く"""

import gym
import tensorflow as tf
import numpy as np

env = gym.make('CartPole-v0')

# # ネットワークのクラス定義
#
# クラスとは、命令や変数をまとめたデータのかたまりです。
#
# １つのクラスオブジェクトに複数のデータを格納し、操作をすることが可能です。


class QNetwork:
    # learning_rateは学習が進まない場合は値を大きくする。進みすぎる場合は値を減らす 　
    def __init__(self, learning_rate=0.01, state_size=4, action_size=2, hidden_size=10, name='QNetwork'):
        with tf.variable_scope(name):
            self.inputs_ = tf.placeholder(
                tf.float32, [None, state_size], name='inputs')
            self.actions_ = tf.placeholder(tf.int32, [None], name='actions')
            one_hot_actions = tf.one_hot(self.actions_, action_size)
            self.targetQs_ = tf.placeholder(tf.float32, [None], name='target')
            self.fc1 = tf.contrib.layers.fully_connected(
                self.inputs_, hidden_size)
            self.fc2 = tf.contrib.layers.fully_connected(self.fc1, hidden_size)
            self.output = tf.contrib.layers.fully_connected(
                self.fc2, action_size, activation_fn=None)
            self.Q = tf.reduce_sum(tf.multiply(
                self.output, one_hot_actions), axis=1)
            self.loss = tf.reduce_mean(tf.square(self.targetQs_ - self.Q))
            self.opt = tf.train.AdamOptimizer(
                learning_rate).minimize(self.loss)


# エクスペリエンス・メモリの定義
from collections import deque


class Memory():
    def __init__(self, max_size=1000):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        idx = np.random.choice(np.arange(len(self.buffer)),
                               size=batch_size, replace=False)
        return [self.buffer[ii] for ii in idx]


# ハイパーパラメーターの定義と初期化
train_episodes = 1000
max_step = 200
gamma = 0.99

explore_start = 1.0
explore_stop = 0.01
decay_rate = 0.0001

hidden_size = 64
learning_rate = 0.0001

memory_size = 10000
batch_size = 20
pretrain_length = batch_size

tf.reset_default_graph()
mainQN = QNetwork(name='main', hidden_size=hidden_size,
                  learning_rate=learning_rate)


# エクスペリエンスメモリーを埋めよう
env.reset()

state, reward, done, _ = env.step(env.action_space.sample())

memory = Memory(max_size=memory_size)

for ii in range(pretrain_length):

    action = env.action_space.sample()
    next_state, reward, done, _ = env.step(action)

    if done:
        next_state = np.zeros(state.shape)
        memory.add((state, action, reward, next_state))

        env.reset()

        state, reward, done, _ = env.step(env.action_space.sample())
    else:
        memory.add((state, action, reward, next_state))
        state = next_state


# トレーニング
saver = tf.train.Saver()
rewards_list = []
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    step = 0
    for ep in range(1, train_episodes):
        total_rewards = 0
        t = 0
        while t < max_step:
            step += 1
            explore_p = explore_stop + \
                (explore_start - explore_stop) * np.exp(-decay_rate * step)
            if explore_p > np.random.rand():
                action = env.action_space.sample()
            else:
                feed = {mainQN.inputs_: state.reshape((1, *state.shape))}
                Qs = sess.run(mainQN.output, feed_dict=feed)
                action = np.argmax(Qs)

            next_state, reward, done, _ = env.step(action)
            total_rewards += reward

            if done:
                next_state = np.zeros(state.shape)
                t = max_step

                # if ep % 100 == 0:
                print('Episode: {}'.format(ep), 'Total Reward: {}'.format(total_rewards),
                      'Total Loss: {:.4f}'.format(loss), 'Explore Prob: {:.4f}'.format(explore_p))

                rewards_list.append((ep, total_rewards))
                memory.add((state, action, reward, next_state))
                env.reset()
                state, reward, done, _ = env.step(env.action_space.sample())

            else:
                memory.add((state, action, reward, next_state))
                state = next_state
                t += 1

            batch = memory.sample(batch_size)
            # print(batch)
            states = np.array([each[0] for each in batch])
            actions = np.array([each[1] for each in batch])
            rewards = np.array([each[2] for each in batch])
            next_states = np.array([each[3] for each in batch])

            target_Qs = sess.run(mainQN.output, feed_dict={
                                 mainQN.inputs_: next_states})
            episode_ends = (next_states == np.zeros(
                state[0].shape)).all(axis=1)
            target_Qs[episode_ends] = (0, 0)

            targets = rewards + gamma * np.max(target_Qs, axis=1)

            loss, _ = sess.run([mainQN.loss, mainQN.opt],
                               feed_dict={mainQN.inputs_: states,
                                          mainQN.targetQs_: targets,
                                          mainQN.actions_: actions})

    saver.save(sess, "checkpoints/cartpole_dqn.ckpt")

# トレーニングの結果を可視化しよう
import matplotlib.pyplot as plt


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / N


eps, rews = np.array(rewards_list).T
smoothed_rews = running_mean(rews, 10)
plt.plot(eps[-len(smoothed_rews):], smoothed_rews)
plt.plot(eps, rews, color='grey', alpha=0.3)
plt.xlabel('Episode')
plt.ylabel('Total Rewards')


test_episodes = 10
test_max_steps = 400
env.reset()
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))

    for ep in range(1, test_episodes):
        t = 0
        while t < test_max_steps:
            env.render()

            feed = {mainQN.inputs_: state.reshape((1, *state.shape))}
            Qs = sess.run(mainQN.output, feed_dict=feed)
            action = np.argmax(Qs)

            next_state, reward, done, _ = env.step(action)

            if done:
                t = test_max_steps
                env.reset()
                state, reward, done, _ = env.step(env.action_space.sample())

            else:
                state = next_state
                t += 1
