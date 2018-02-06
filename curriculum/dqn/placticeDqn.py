import gym

# カーネルがクラッシュする
# env  = gym.make("CartPole-v0")
# env.reset()
# for _ in range(1000):
#     env.render()
#     env.step(env.action_space.sample())

env = gym.make("CartPole-v0")
# 20回でランダムに値を与えて倒れないように検証する
for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("このエピソードは{}回目で終了しました".format(t+1))
            break
