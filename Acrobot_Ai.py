import gymnasium as gym
import numpy as np
import tqdm as tqdm
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patch
from collections import defaultdict

# env = gym.make("Acrobot-v1", render_mode = "rgb_array")
env = gym.make("Acrobot-v1")
pi = 3.14
initial_time = time.time()
class Ai_agent():
    def __init__(self , learning_rate:float , initial_epsilon:float , epsilon_decay:float , final_epsilon:float , discount_factor:float ):
        self.Q_value = defaultdict(lambda : np.zeros(env.action_space.n))
        self.lr = learning_rate
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.discount_factor = discount_factor
        self.win_rate = []

    def get_action(self , obs: tuple[float, float , float  ,float,float,float]) -> int:
        if (np.random.random() < self.epsilon):
            return env.action_space.sample()
        else:
            return int(np.argmax(self.Q_value[obs]))

    def update(self ,obs: tuple[float, float , float  ,float,float,float] , action: float,reward : float , next_obs: tuple[float, float , float  ,float,float,float] , terminated : bool ):
        if terminated:
            target = reward
        else:
            target = reward + self.discount_factor* np.max(self.Q_value[next_obs])

        temporal_diff = target - self.Q_value[obs][action]
        self.Q_value[obs][action] += self.lr* temporal_diff
        # self.training_error.append(temporal_diff)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon , self.epsilon - self.epsilon_decay)

def observation_correction(obs: tuple[float, float , float  ,float,float,float]) -> tuple[float, float , float  ,float,float,float]:
    for i in range(4):
        obs[i] = int(obs[i]/0.1)

    obs[4] = int(obs[4]/pi)
    obs[5] = int(obs[5]/pi)

    return tuple(obs)

agent = Ai_agent(0.15,1,0.00018,0.1,0.95)
n_episodes = 5000
env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length= n_episodes)

for episode in tqdm.tqdm(range(1 , n_episodes + 1)):
    episode_end = False
    observation , info = env.reset()

    while not episode_end:

        corrected_observation = observation_correction(observation.copy())
        action = agent.get_action(corrected_observation)
        next_observation , reward , terminated , truncated , info = env.step(action)

        corrected_next_observation = observation_correction(next_observation.copy())

        agent.update(corrected_observation , action , reward , corrected_next_observation , terminated)

        observation = (next_observation)

        episode_end = terminated or truncated

    if not (episode % (n_episodes/10) ):
        print(f"Time Taken for next 10% : {time.time()- initial_time}")
        initial_time = time.time()
    
    agent.win_rate.append(int(terminated))

    if episode<4000:
        agent.decay_epsilon()
    else:
        agent.epsilon = 0


data_group = 1000
figure , axs = plt.subplots(ncols = 3 , figsize = (12,3 ))
axs[0].set_title("Reward Distribution")
reward_moving_average = (np.convolve(np.array(env.return_queue).flatten() , np.ones(data_group) , mode = "valid"))/data_group
axs[0].plot(range(len(reward_moving_average)) , reward_moving_average)

axs[1].set_title("Length Distribution")
length_moving_average = (np.convolve(np.array(env.length_queue).flatten() , np.ones(data_group) , mode = "valid"))/data_group
axs[1].plot(range(len(length_moving_average)) , length_moving_average)

# axs[2].set_title("Training error")
# Training_error_average = (np.convolve(np.abs(np.array(agent.training_error)) , np.ones(data_group) , mode = "same"))/data_group
# axs[2].plot(range(len(Training_error_average)) , Training_error_average)

axs[2].set_title("Win rate")
win = np.convolve(np.array(agent.win_rate) ,np.ones(data_group) , mode = "valid" )/data_group*100
axs[2].plot(range(len(win)) , win)

plt.tight_layout()
plt.show()

print(f"Q Table Size :{len(agent.Q_value)}")


record_env = gym.make("Acrobot-v1", render_mode="rgb_array")

record_env = gym.wrappers.RecordVideo(
    record_env,
    video_folder="Acrobot_Videos",
    name_prefix="AI_playing",
    episode_trigger=lambda episode: True
)

n_test_episodes = 5

for episode in range(n_test_episodes):

    observation, info = record_env.reset()
    episode_end = False

    while not episode_end:

        corrected_observation = observation_correction(observation.copy())

        # Pure exploitation: choose the best learned action
        action = int(np.argmax(agent.Q_value[corrected_observation]))

        next_observation, reward, terminated, truncated, info = record_env.step(action)

        observation = next_observation
        episode_end = terminated or truncated

    print(
        f"Episode {episode + 1}: "
        f"{'SUCCESS' if terminated else 'FAILED'}"
    )

record_env.close()