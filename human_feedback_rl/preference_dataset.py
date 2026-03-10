import torch


class PreferenceDataset:

    def __init__(self):
        self.data = []

    def add(self, item1, item2, probs):
        self.data.append((item1, item2, probs))

    def sample(self, batch_size):
        idx = torch.randint(0, len(self.data), (batch_size,))
        return [self.data[i] for i in idx]