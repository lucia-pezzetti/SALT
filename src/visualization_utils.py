import os
from typing import Sequence
import matplotlib.pyplot as plt

class TrainingLogger:
    """
    Logger for training metrics: total rewards and wait times per episode.
    """
    def __init__(self):
        # Lists to store metrics per episode
        self.episode_rewards = []  # Total reward per episode
        self.avg_waits = []        # Average wait time per episode

    def log(self, reward: float, waits: Sequence[float]):
        """
        Log the total reward and wait times for an episode.

        Args:
            reward: Total reward accumulated in the episode.
            waits: Sequence of wait times recorded at each step.
        """
        # Record total reward
        self.episode_rewards.append(reward)
        # Compute and record average wait time
        if waits:
            avg_wait = float(sum(waits) / len(waits))
        else:
            avg_wait = 0.0
        self.avg_waits.append(avg_wait)

    def save_plots(self, out_dir: str = "plots"):
        """
        Save training plots for rewards and average wait times.

        Args:
            out_dir: Directory where plots will be saved. Created if nonexistent.
        """
        # Ensure output directory exists
        os.makedirs(out_dir, exist_ok=True)

        # Plot: Reward per Episode
        plt.figure()
        plt.plot(self.episode_rewards)
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("Reward per Episode")
        plt.grid(True)
        reward_path = os.path.join(out_dir, "reward_per_episode.png")
        plt.savefig(reward_path)
        plt.close()

        # Plot: Average Wait Time per Episode
        plt.figure()
        plt.plot(self.avg_waits)
        plt.xlabel("Episode")
        plt.ylabel("Average Wait Time")
        plt.title("Average Wait per Episode")
        plt.grid(True)
        wait_path = os.path.join(out_dir, "avg_wait_per_episode.png")
        plt.savefig(wait_path)
        plt.close()

