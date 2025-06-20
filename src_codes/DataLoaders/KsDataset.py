from torch.utils.data import Dataset
import torch

class KSDataset(Dataset):
    def __init__(self, ks_array, seq_length=1):
        self.inputs = []
        self.targets = []

        num_experiments, num_sims, time_steps, _,_,_ = ks_array.shape

        for exp in range(num_experiments):
            for sim in range(num_sims):
                for t in range(time_steps - seq_length):
                    # Sequence: (seq_length, spatial_dim)
                    x_seq = ks_array[exp, sim, t:t+seq_length, 0, :]
                    y_target = ks_array[exp, sim, t + seq_length, 0, :]

                    self.inputs.append(torch.tensor(x_seq, dtype=torch.float32))
                    self.targets.append(torch.tensor(y_target, dtype=torch.float32))

        self.inputs = torch.stack(self.inputs)    # (N, seq_length, spatial_dim)
        self.targets = torch.stack(self.targets)  # (N, spatial_dim)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]