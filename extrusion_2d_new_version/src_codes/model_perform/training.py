import torch
from tqdm import tqdm
import os
from .validation_predictions import validation_prediction

def training_loop(model, device, criterion, optimizer, scheduler, train_loaders, val_loaders, epochs,
                  len_train_dataset, len_val_dataset, storing_path, res_path, preds_path, val_samples):

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        loaders_iters = [iter(loader) for loader in train_loaders]
        steps = min(len(loader) for loader in train_loaders)

        print(f"Epoch {epoch+1}/{epochs} | Steps per loader: {steps}")

        for _ in tqdm(range(steps), desc="Training"):
            batch_outputs = []
            batch_targets = []

            for loader_iter in loaders_iters:
                # try:
                #     inputs, targets, ts, pde_params = next(loader_iter)
                # except StopIteration:
                #     continue
                inputs, targets, ts, pde_params = next(loader_iter)
                inputs = inputs.squeeze(0).to(device)
                targets = targets.squeeze(0).to(device)
                ts = ts.to(device)
                pde_params = pde_params.to(device)
                pde_ts = torch.cat([pde_params.squeeze(0), ts], dim=0)

                outputs = model(inputs, pde_ts)
                batch_outputs.append(outputs)
                batch_targets.append(targets)

            total_outputs = torch.cat(batch_outputs, dim=0)
            total_targets = torch.cat(batch_targets, dim=0)

            loss = criterion(total_outputs, total_targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * total_outputs.size(0)

        avg_train_loss = train_loss / len_train_dataset
        train_losses.append(avg_train_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0

        val_loaders_iters = [iter(loader) for loader in val_loaders]
        val_steps = min(len(loader) for loader in val_loaders)

        print(f"Validation Steps per loader: {val_steps}")

        with torch.no_grad():
            for _ in tqdm(range(val_steps), desc="Validation"):
                batch_outputs = []
                batch_targets = []

                for val_iter in val_loaders_iters:
                    # try:
                    #     inputs, targets, ts, pde_params = next(val_iter)
                    # except StopIteration:
                    #     continue
                    inputs, targets, ts, pde_params = next(val_iter)
                    inputs = inputs.squeeze(0).to(device)
                    targets = targets.squeeze(0).to(device)
                    ts = ts.to(device)
                    pde_params = pde_params.to(device)
                    pde_ts = torch.cat([pde_params.squeeze(0), ts], dim=0) 

                    outputs = model(inputs, pde_ts)
                    batch_outputs.append(outputs)
                    batch_targets.append(targets)

                total_outputs = torch.cat(batch_outputs, dim=0)
                total_targets = torch.cat(batch_targets, dim=0)

                loss = criterion(total_outputs, total_targets)
                val_loss += loss.item() * total_outputs.size(0)

        avg_val_loss = val_loss / len_val_dataset
        val_losses.append(avg_val_loss)

        # Logging
        log_line = f"Epoch {epoch+1}/{epochs} | Train MSE: {avg_train_loss:.6f} | Val MSE: {avg_val_loss:.6f}"
        print(log_line)
        with open(os.path.join(res_path, 'training_log.txt'), "a") as f:
            f.write(log_line + "\n")

        # Periodic checkpoint
        if (epoch + 1) % 5 == 0:
            save_path = os.path.join(storing_path, f"model_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss
            }, save_path)
            print(f"Saved checkpoint at epoch {epoch+1} to {save_path}")
            validation_prediction(model, device, val_samples,preds_path, epoch+1)

        # Save best checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss
            }, os.path.join(storing_path, 'best_model_checkpoint.pth'))
            print(f"Saved BEST checkpoint at epoch {epoch+1}")
            with open(os.path.join(res_path, 'training_log.txt'), "a") as f:
                f.write(f"Saved BEST checkpoint at epoch {epoch+1}\n")

        if scheduler is not None:
            scheduler.step()

        print("\n")
        

    return train_losses, val_losses
