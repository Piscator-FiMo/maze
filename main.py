from itertools import count
import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
import math
import random
import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from GridWorldEnv import GridWorldEnv
from Labyrinth import Labyrinth
from memory import ReplayMemory, Transition
from model import DQN


def transform_to_one_hot_vector(n, n_observations):
    one_hot_1 = np.zeros(n_observations)
    one_hot_1[n] = 1
    return torch.tensor(one_hot_1.ravel(), dtype=torch.float32, device=device).unsqueeze(0)


def select_action(state, policy_net, steps_done):
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)


def optimize_model(memory, policy_net, target_net, optimizer):
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)),
                                  device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


def train(env, episodes, folder="labyrinth-training"):
    n_observations = env.observation_space.n
    n_actions = env.action_space.n

    policy_net = DQN(n_observations, n_actions).to(device)
    target_net = DQN(n_observations, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)

    memory = ReplayMemory(MEMORY_CAPACITY)

    env = RecordVideo(env, video_folder=folder, name_prefix="labyrinth",
                      episode_trigger=lambda x: x % 100 == 0 or x >= episodes - 10, fps=12)
    env = RecordEpisodeStatistics(env, buffer_length=episodes)
    for _ in range(episodes):
        # Initialize the environment and get its state
        options = {"render_mode": "rgb_array"}
        state, _ = env.reset(options=options)
        state = transform_to_one_hot_vector(state, n_observations)
        done = False
        for step in count():
            if done:
                break
            action = select_action(state, policy_net, step)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = transform_to_one_hot_vector(observation, n_observations)

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            optimize_model(memory=memory, policy_net=policy_net, target_net=target_net, optimizer=optimizer)

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
            target_net.load_state_dict(target_net_state_dict)

        print(f"Episode {env.episode_count}/{episodes} {"completed" if terminated else "truncated"}: steps={
              env.episode_lengths} reward={env.episode_returns}")

    env.close()

    # visualize the episode rewards, episode length and training error in one figure
    _, axs = plt.subplots(1, 3, figsize=(20, 8))

    # np.convolve will compute the rolling mean for 100 episodes

    axs[0].plot(np.convolve(env.return_queue, np.ones(100)))
    axs[0].set_title("Episode Rewards")
    axs[0].set_xlabel("Episode")
    axs[0].set_ylabel("Reward")

    axs[1].plot(np.convolve(env.length_queue, np.ones(100)))
    axs[1].set_title("Episode Lengths")
    axs[1].set_xlabel("Episode")
    axs[1].set_ylabel("Length")

    axs[2].plot(np.convolve(env.time_queue, np.ones(100)))
    axs[2].set_title("Episode Times")
    axs[2].set_xlabel("Episode")
    axs[2].set_ylabel("Time")

    plt.tight_layout()
    # Save plots
    plt.savefig(f"{folder}/episodes_visualization.png")

    policy_file = f"{folder}/policy_net.pt"
    torch.save(policy_net.state_dict(), policy_file)
    return policy_file


def test(env, policy_file, episodes):
    n_observations = env.observation_space.n

    policy_dqn = DQN(n_observations, env.action_space.n).to(device)
    policy_dqn.load_state_dict(torch.load(policy_file))
    policy_dqn.eval()

    for _ in range(episodes):
        state, _ = env.reset(options={"render_mode": "human"})
        done = False
        while not done:
            with torch.no_grad():
                action = policy_dqn(transform_to_one_hot_vector(state, n_observations)).argmax().item()

            state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

    env.close()


if __name__ == "__main__":
    # set up matplotlib
    is_ipython = 'inline' in matplotlib.get_backend()
    if is_ipython:
        from IPython import display

    # if GPU is to be used
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )
    print("Device:", device)

    gym.register(id="gymnasium_env/GridWorld-v0", entry_point=GridWorldEnv)

    # BATCH_SIZE is the number of transitions sampled from the replay buffer
    # GAMMA is the discount factor as mentioned in the previous section
    # EPS_START is the starting value of epsilon
    # EPS_END is the final value of epsilon
    # EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
    # TAU is the update rate of the target network
    # LR is the learning rate of the ``AdamW`` optimizer
    BATCH_SIZE = 128
    GAMMA = 0.99
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 900
    TAU = 0.005
    LR = 1e-4
    MEMORY_CAPACITY = 10_000

    n_train_episodes = 600 if str(device) == "cpu" else 1_000
    n_test_episodes = 10

    labyrinth = Labyrinth(10, 10, seed=42)
    env = gym.make("gymnasium_env/GridWorld-v0", labyrinth=labyrinth)
    file = train(env, episodes=n_train_episodes)
    # for _ in range(2000):
    #     labyrinth.regenerate_start()
    # file = "big-labyrinth/policy_net.pt"
    test(env, file, n_test_episodes)
