import torch
from torch.utils.data import Dataset
from collections import defaultdict
import numpy as np

class SubGroupDataset(Dataset):
    def __init__(self, data, pde_param, batch_size=16):
        """
        data: numpy array (IC, TS, 1, 160, 160)
        pde_param: numpy array (3,)
        """
        self.pde_param = torch.tensor(pde_param, dtype=torch.float32)
        self.batch_size = batch_size
        self.samples = []  # List of (inputs, targets, ts)

        # Generate valid (ic, ts) pairs (ts < 42)
        pairs = [(ic, ts) for ic in range(data.shape[0]) for ts in range(data.shape[1] - 1)]

        # Group by time step
        ts_to_ics = defaultdict(list)
        for ic, ts in pairs:
            ts_to_ics[ts].append(ic)

        # Chunk each ts group into batches
        for ts, ic_list in ts_to_ics.items():
            for i in range(0, len(ic_list), self.batch_size):
                chunk = ic_list[i:i + self.batch_size]
                input_batch = np.stack([data[ic, ts] for ic in chunk])       # shape (B,1,160,160)
                target_batch = np.stack([data[ic, ts + 1] for ic in chunk])  # shape (B,1,160,160)

                input_tensor = torch.tensor(input_batch, dtype=torch.float32)
                target_tensor = torch.tensor(target_batch, dtype=torch.float32)

                self.samples.append((input_tensor, target_tensor, ts))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        inputs, targets, ts = self.samples[idx]
        return inputs, targets, ts, self.pde_param
