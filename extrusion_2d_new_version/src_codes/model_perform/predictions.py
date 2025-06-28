from torch.utils.data import Dataset
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from torch.utils.data import DataLoader

class KSTrajectoryDataset(Dataset):
    def __init__(self, ks_array):
        self.inputs = []

        num_experiments, num_sims, time_steps, _, _,_ = ks_array.shape

        for exp in range(num_experiments):
            for sim in range(num_sims):
                x_seq = ks_array[exp, sim, :, 0, :, :]  # shape: (time_steps, spatial_dim)
                self.inputs.append(torch.tensor(x_seq, dtype=torch.float32))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx]  # shape: (time_steps, spatial_dim)

def vanilla_preds(ks_test, model, device):
    test_dataset = KSTrajectoryDataset(ks_test)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    total_predictions = []
    total_targets = []

    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            traj = batch[0].to(device)  # shape: (T, H, W)
            time_steps, H, W = traj.shape

            predictions = []
            input_ar = traj[0].unsqueeze(0).unsqueeze(0)  # shape: (1, 1, H, W)

            for t in range(time_steps - 1):
                output = model(input_ar)  # expected shape: (1, 1, H, W)
                output = output.squeeze(0).squeeze(0)

                if (output == 0).all():
                    print("Output is all zeros at time step:", t)

                predictions.append(output.cpu().numpy())
                input_ar = output.unsqueeze(0).unsqueeze(0).detach()

            predictions = np.stack(predictions)     # shape: (T-1, H, W)
            total_predictions.append(predictions)
            targets = traj[1:].cpu().numpy()
            total_targets.append(targets)  

    total_predictions = np.stack(total_predictions)  # shape: (num_samples, T-1, H, W)
    total_targets = np.stack(total_targets)  # shape: (num_samples, T-1, H, W)
    print("Total predictions shape:", total_predictions.shape)
    print("Total targets shape:", total_targets.shape)
    return total_predictions, total_targets



