import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

def training_loop(model,device, criterion, optimizer,schedular, train_loader, val_loader, epochs, len_train_dataset, len_val_dataset, storing_path, res_path, best_model_path):

    train_losses = []
    val_losses = []

    best_val_loss  = float('inf')

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for inputs, targets in tqdm(train_loader, desc=f"[Train Epoch {epoch+1}/{epochs}]"):
            inputs, targets = inputs.to(device), targets.to(device)
            output = model(inputs)
            output = output.squeeze(1)
            loss = criterion(output, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.detach().cpu().item() * inputs.size(0)
            inputs.detach()
            targets.detach()

        avg_train_loss = train_loss / len_train_dataset
        train_losses.append(avg_train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc=f"[Val Epoch {epoch+1}/{epochs}]", leave=False):
                inputs, targets = inputs.to(device), targets.to(device)
                output = model(inputs)
                output = output.squeeze(1)
                loss = criterion(output, targets)
                val_loss += loss.detach().cpu().item() * inputs.size(0)
                inputs.detach()
                targets.detach()

        avg_val_loss = val_loss / len_val_dataset
        val_losses.append(avg_val_loss)

        log_line = f"Epoch {epoch+1}/{epochs} | Train RMSE: {avg_train_loss:.6f} | Val RMSE: {avg_val_loss:.6f}"
        print(log_line)
        with open(os.path.join(res_path,'training_log.txt'), "a") as f:
            f.write(log_line + "\n")
        
        if (epoch + 1) % 10 == 0:
            save_path = os.path.join(storing_path, f"model_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss
            }, save_path)
            print(f"Saved checkpoint at epoch {epoch+1} to {save_path}")
        print(f"\n")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss
            }, os.path.join(best_model_path, 'best_model_checkpoint.pth'))
            print(f" Saved BEST checkpoint at epoch {epoch+1} to {best_model_path}")
            with open(os.path.join(res_path, 'training_log.txt'), "a") as f:
                f.write(f"Saved BEST checkpoint at epoch {epoch+1} to {best_model_path}\n")

        if schedular is not None:
            schedular.step()

    return train_losses, val_losses
