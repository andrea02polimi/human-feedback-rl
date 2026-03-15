import torch

from human_feedback_rl.training.base_trainer import BaseTrainer


class PreferenceTrainer(BaseTrainer):

    def __init__(self, reward_model, optimizer, dataset, run_dir=None):

        super().__init__(
            env=None,
            policy=None,
            expert_model=None,
            optimizer=optimizer,
            run_dir=run_dir,
            name="preferences"
        )

        self.reward_model = reward_model
        self.dataset = dataset

    # ------------------------------------------------

    def train(self, epochs=2000, batch_size=64):

        for epoch in range(1, epochs + 1):

            batch = self.dataset.sample(batch_size)

            loss_total = 0

            for step1, step2, probs in batch:

                s1 = step1.state.unsqueeze(0)
                a1 = torch.tensor([step1.action])

                s2 = step2.state.unsqueeze(0)
                a2 = torch.tensor([step2.action])

                r1 = self.reward_model(s1, a1)
                r2 = self.reward_model(s2, a2)

                logits = torch.cat([r1, r2], dim=1)

                target = torch.tensor(probs.value).unsqueeze(0)

                loss = -(target * torch.log_softmax(logits, dim=1)).sum()

                loss_total += loss

            loss_total /= batch_size

            self.optimizer.zero_grad()
            loss_total.backward()
            self.optimizer.step()

            # tensorboard logging via BaseTrainer
            self.log_episode(
                epoch,
                reward=0,                 # non c'è reward qui
                length=batch_size,
                loss=loss_total.item(),
                kl=0,
                match=0,
                entropy=0
            )

            if epoch % 100 == 0:

                print(
                    f"Preference epoch {epoch} | "
                    f"loss {loss_total.item():.4f}"
                )