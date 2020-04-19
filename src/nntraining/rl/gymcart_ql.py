import time
import collections
import gym
import sklearn
import numpy as np
import operator
import pprint

rng = np.random.default_rng()


def get_state(obs, mins, maxs):
    return tuple((10 * ((obs - mins) / (maxs - mins))).astype(int))


def update_minmax(obs, mins, maxs):
    beef = np.concatenate((obs, mins, maxs)).reshape(-1, 4)
    return beef.min(axis=0), beef.max(axis=0)


def policy(state, values, action_space):
    actions = list(range(action_space.n))
    choices = [(action, values[(state, action)]) for action in actions]
    action_max = sorted(choices, key=operator.itemgetter(1), reverse=True)[0][0]
    return action_max if rng.random(1) < .5 else action_space.sample()


alpha = .01
gamma = .05
values = collections.defaultdict(lambda: 0)

history = []

mins = np.array([0, 0, 0, 0])
maxs = np.array([1, 1, 1, 1])

lengths = []
env = gym.make('CartPole-v0')
for i_episode in range(1000):
    # init
    observation = env.reset()
    state = get_state(observation, *update_minmax(observation, mins, maxs))
    for t in range(100):
        # env.render()
        # time.sleep(.01)

        # compute
        action = policy(state, values, env.action_space)
        observation, reward, done, info = env.step(action)

        # update value
        next_state = get_state(observation, *update_minmax(observation, mins, maxs))
        actions = list(range(env.action_space.n))
        next_values = [values[(next_state, action)] for action in actions]
        values[(state, action)] += \
            alpha * (reward + gamma * max(next_values) - values[state])
        state = next_state

        if done:
            lengths.append(t+1)
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()

size = 10
buckets = [size * (l // size) for l in lengths]
print()
print(collections.Counter(buckets).most_common(20))
pprint.pprint(list(sorted(values.items(), key=operator.itemgetter(1),
                          reverse=True))[:20])
