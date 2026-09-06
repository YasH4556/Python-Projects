import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patch
import seaborn as sns
from collections import defaultdict
from tqdm import tqdm
import random
from IPython.display import clear_output
import seaborn

env = gym.make("Blackjack-v1",sab = True , render_mode = "rgb_array")

class BlackjackAgent():
    def __init__(self,learning_rate: float, initial_epsilon:float , epsilon_decay:float , final_epsilon:float , discount_factor:float = 0.95 , ):
        self.Q_value = defaultdict(lambda : np.zeros(env.action_space.n))
        self.lr = learning_rate
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon =final_epsilon
        self.discount_factor = discount_factor

        self.training_error = []
        self.win_rate = []

    def get_action(self, obs:tuple[float,float,bool]) -> int:

        if np.random.random() < self.epsilon:
            return env.action_space.sample()
        else:
            return int(np.argmax(self.Q_value[obs]))

    def update(self,
               obs:tuple[float,float,bool],action:int, reward:float,Next_obs:tuple[float,float,bool],terminated:bool):
        # next_q_value = np.max(self.Q_value[Next_obs])
        # temporal_diff = reward + self.discount_factor*next_q_value - self.Q_value[obs][action]
        # self.Q_value[obs][action] += self.lr*temporal_diff
        if terminated:
            target = reward
            self.win_rate.append(int(reward>0))
        else:
            target = reward + self.discount_factor * np.max(self.Q_value[Next_obs])

        temporal_diff = target - self.Q_value[obs][action]

        self.Q_value[obs][action] += self.lr * temporal_diff
        self.training_error.append(temporal_diff)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

n_episodes = 5
agent = BlackjackAgent(0.01 , 1.0 , 0.0001 , 0.1)
env = gym.wrappers.RecordEpisodeStatistics(env , buffer_length = n_episodes)

for episode in tqdm(range(n_episodes)):
    episode_done = False
    observation , info = env.reset()
    monte_episode = []

    while not episode_done:
        action = agent.get_action(observation)
        next_observation , reward, terminated , truncated, info = env.step(action)
        agent.update(observation,action,reward,next_observation , terminated)

        # frame = env.render()
        # plt.imshow(frame)
        # print(type(frame))
        # print(frame)
        observation = next_observation
        episode_done = terminated or truncated


    agent.decay_epsilon()

rolling_length = 1000
fig, axs = plt.subplots(ncols=4, figsize=(12, 5))
axs[0].set_title("Episode rewards")
# compute and assign a rolling average of the data to provide a smoother graph
reward_moving_average = (
    np.convolve(
        np.array(env.return_queue).flatten(), np.ones(rolling_length), mode="valid"
    )
    / rolling_length
)
axs[0].plot(range(len(reward_moving_average)), reward_moving_average)
axs[1].set_title("Episode lengths")
length_moving_average = (
    np.convolve(
        np.array(env.length_queue).flatten(), np.ones(rolling_length), mode="same"
    )
    / rolling_length
)
axs[1].plot(range(len(length_moving_average)), length_moving_average)
axs[2].set_title("Training Error")
training_error_moving_average = (
    np.convolve(np.abs(np.array(agent.training_error)), np.ones(rolling_length), mode="same")
    / rolling_length
)
axs[2].plot(range(len(training_error_moving_average)), training_error_moving_average)

axs[3].set_title("Win Rate")
win_rate = (np.convolve(np.array(agent.win_rate), np.ones(rolling_length), mode = "valid"))/rolling_length*100
axs[3].plot(range(len(win_rate)), win_rate)
plt.tight_layout()
plt.show()