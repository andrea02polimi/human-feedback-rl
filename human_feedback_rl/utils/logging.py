import os
import datetime
from torch.utils.tensorboard import SummaryWriter


class Logger:

    def __init__(self, base_dir="tensorboard"):

        path = os.path.join(base_dir)

        os.makedirs(path, exist_ok=True)

        self.writer = SummaryWriter(path)

    def log_episode(self, episode, reward, length, loss, kl, match, entropy):

        # -------- terminal output --------

        print(
            f"episode {episode:4d} | "
            f"reward {reward:8.2f} | "
            f"length {length:4d} | "
            f"loss {loss:8.4f} | "
            f"match {match:6.3f} | "
            f"entropy {entropy:6.3f} | "
            f"kl {kl:6.3f}"
        )

        # -------- tensorboard --------

        self.writer.add_scalar("episode/reward", reward, episode)
        self.writer.add_scalar("episode/length", length, episode)
        self.writer.add_scalar("episode/loss", loss, episode)

        self.writer.add_scalar("policy/kl_expert_agent", kl, episode)
        self.writer.add_scalar("policy/action_match", match, episode)
        self.writer.add_scalar("policy/entropy", entropy, episode)