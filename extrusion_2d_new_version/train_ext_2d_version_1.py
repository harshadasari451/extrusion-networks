import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
import pickle

import matplotlib.pyplot as plt
import os

from src_codes.models.primary_func import PrimaryNetwork
from src_codes.DataLoaders.KsDataset import SubGroupDataset
from src_codes.model_perform.training import training_loop

data_storing_path = '/home/CAMPUS/hdasari/apebench_experiments/new_mse_experiments/extrusion_2d_new_version/checkpoints/version_1_june27'
results_storing_path = '/home/CAMPUS/hdasari/apebench_experiments/new_mse_experiments/extrusion_2d_new_version/results/version_1_june27'
preds_path = '/home/CAMPUS/hdasari/apebench_experiments/new_mse_experiments/extrusion_2d_new_version/preds_and_targets/version_1_june27'

os.makedirs(data_storing_path, exist_ok=True)
os.makedirs(results_storing_path, exist_ok=True)
os.makedirs(preds_path, exist_ok=True)


pde = np.array([[-1, -15, -6], [-1.6, -15, -6], [-1.8, -15, -6]])
exp1_data = np.load('/home/CAMPUS/hdasari/apebench_experiments/ks_2d/data/KS_2d_train_data_exp1.npy')
exp2_data = np.load('/home/CAMPUS/hdasari/apebench_experiments/ks_2d/data/KS_2d_train_data_exp4.npy')
exp3_data = np.load('/home/CAMPUS/hdasari/apebench_experiments/ks_2d/data/KS_2d_train_data_exp5.npy')

val_exp1 = np.load('/home/CAMPUS/hdasari/apebench_experiments/ks_2d/data/KS_2d_test_data_exp1.npy')
val_exp2 = np.load('/home/CAMPUS/hdasari/apebench_experiments/ks_2d/data/KS_2d_test_data_exp4.npy')
val_exp3 = np.load('/home/CAMPUS/hdasari/apebench_experiments/ks_2d/data/KS_2d_test_data_exp5.npy')

val_samples = np.load('/home/CAMPUS/hdasari/apebench_experiments/ks_2d/data/validation_samples.npy')

print(exp1_data.shape, exp2_data.shape, exp3_data.shape)

exp1_dataset = SubGroupDataset(exp1_data, pde[0], batch_size=32)
exp2_dataset = SubGroupDataset(exp2_data, pde[1], batch_size=32)
exp3_dataset = SubGroupDataset(exp3_data, pde[2], batch_size=32)

# val_pde = np.array([[-1.2,-15,-6], [ -1.4, -15, -6]])
val_exp1_dataset = SubGroupDataset(val_exp1,pde[0])
val_exp2_dataset = SubGroupDataset(val_exp2, pde[1])
val_exp3_dataset = SubGroupDataset(val_exp3, pde[2])

exp1_loader = DataLoader(exp1_dataset, batch_size=1, shuffle=True, num_workers=4)
exp2_loader = DataLoader(exp2_dataset, batch_size=1, shuffle = True, num_workers=4)
exp3_loader = DataLoader(exp2_dataset, batch_size=1, shuffle = True, num_workers=4)

val_exp1 = DataLoader(val_exp1_dataset, batch_size=1, shuffle=False)
val_exp2 = DataLoader(val_exp2_dataset, batch_size=1, shuffle = False)
val_exp3 = DataLoader(val_exp3_dataset, batch_size=1, shuffle = False)

train_loaders = [exp1_loader, exp2_loader, exp3_loader]
len_train_dataset = len(exp1_dataset)+ len(exp2_dataset)+ len(exp3_dataset)

val_loaders = [val_exp1, val_exp2]
len_val_dataset = len(val_exp1_dataset) + len(val_exp2_dataset) + len(val_exp3_dataset)


unet_1d_weights_path = '/home/CAMPUS/hdasari/apebench_experiments/mse_experiments/vanilla_1d/checkpoints/new_june19_mse_epoch_20_unet_1d_weights_biases.pth'


device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = PrimaryNetwork(unet_1d_weights_path=unet_1d_weights_path, device=device).to(device)
criterion = nn.MSELoss(reduction='mean')
optimizer = optim.Adam(model.parameters(), lr=1e-3)
epochs = 30


with open(os.path.join(results_storing_path, 'training_log.txt'), "a") as f:
    f.write(f"{'='*50}\n")
    f.write(f"Training data samples : {exp1_data.shape[0]}")
    f.write(f"Model: {model.__class__.__name__}\n")
    f.write(f"Device: {device}\n")
    f.write(f"Optimizer: {optimizer.__class__.__name__}\n")
    f.write(f"Loss Function: {criterion.__class__.__name__}\n")
    f.write(f"Learning Rate: {optimizer.defaults['lr']}\n")
    f.write(f"Weight Decay: {optimizer.defaults['weight_decay']}\n")
    f.write(f"Epochs: {epochs}\n")
    f.write(f"unet_1d_checkpoint: {unet_1d_weights_path}\n")
    f.write(f"{'-'*50}\n")

train_losses, val_losses = training_loop(model,device, criterion, optimizer, None,train_loaders, val_loaders, epochs, len_train_dataset, len_val_dataset, data_storing_path, results_storing_path, preds_path, val_samples)

with open(os.path.join(results_storing_path , 'train_losses.pkl'), 'wb') as f:
    pickle.dump((train_losses, val_losses), f)

with open(os.path.join(results_storing_path, 'training_log.txt'), "a") as f:
    f.write("Training completed\n")

# Plot losses
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss', linewidth=2)
plt.plot(val_losses, label='Validation Loss', linewidth=2)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Losses')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(results_storing_path, 'loss_plot.png'))
plt.close()


print("extrusion unet 2d scuessfully trainined and tested")