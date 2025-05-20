import os
import matplotlib.pyplot as plt
from typing import List

class TrainingLogger:
    def __init__(self):
        # existing buffers
        self.episode_rewards: List[float] = []
        self.episode_waits:    List[List[float]] = []
        # new metric buffers
        self.loss_history:     List[float] = []
        # additional metric buffers for DQN convergence
        self.update_norm_history:  List[float] = []
        self.q_stability_history:  List[float] = []

    def log(self, reward: float, waits: List[float]):
        """
        Log per-episode reward and wait times.
        """
        self.episode_rewards.append(reward)
        self.episode_waits.append(waits)

    def log_metrics(
        self,
        losses: List[float],
        update_norms: List[float] = None,
        q_stabilities: List[float] = None
    ):
        """
        Log training metrics: per-epoch (or per-step) loss, Q-value spread,
        network update norm, and Q-value stability.
        """
        self.loss_history = losses
        if update_norms is not None:
            self.update_norm_history = update_norms
        if q_stabilities is not None:
            self.q_stability_history = q_stabilities

    def save_plots(self, out_dir: str = "plots"):
        """
        Save reward and wait-time plots, plus any metric plots if logged.
        """
        os.makedirs(out_dir, exist_ok=True)

        # 1) Reward per episode
        plt.figure()
        plt.plot(self.episode_rewards)
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Reward per Episode')
        plt.grid(True)
        plt.savefig(os.path.join(out_dir, 'reward_per_episode.png'))
        plt.close()

        # 2) Average wait per episode
        avg_waits = [sum(w)/len(w) if w else 0.0 for w in self.episode_waits]
        plt.figure()
        plt.plot(avg_waits)
        plt.xlabel('Episode')
        plt.ylabel('Average Wait Time')
        plt.title('Average Wait Time per Episode')
        plt.grid(True)
        plt.savefig(os.path.join(out_dir, 'avg_wait_per_episode.png'))
        plt.close()

        # 3) Training metrics, if available
        if self.loss_history:
            plt.figure()
            plt.plot(self.loss_history, label='Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Value')
            plt.title('Training Loss per Epoch')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(out_dir, 'loss.png'))
            plt.close()

        # 4) DQN convergence metrics
        if self.update_norm_history:
            plt.figure()
            plt.plot(self.update_norm_history, label='Update Norm')
            plt.xlabel('Epoch')
            plt.ylabel('||Δθ||')
            plt.title('Network Update Norm per Epoch')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(out_dir, 'update_norm.png'))
            plt.close()

        if self.q_stability_history:
            plt.figure()
            plt.plot(self.q_stability_history, label='Q-Value Stability')
            plt.xlabel('Epoch')
            plt.ylabel('Mean |ΔQ|')
            plt.title('Q-Value Stability per Epoch')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(out_dir, 'q_value_stability.png'))
            plt.close()
