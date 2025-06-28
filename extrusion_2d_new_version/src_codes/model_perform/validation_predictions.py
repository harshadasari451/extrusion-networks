import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import torch
import numpy as np

def validation_prediction(model, device, val_samples, preds_path, epoch):
    pde_params = [[-1,-15,-6],[-1.2,-15,-6],[-1.4,-15,-6],[-1.6,-15,-6],[-1.8,-15,-6],[-2,-15,-6]]
    
    with torch.no_grad():
        epoch_name = f"{epoch}_epoch"
        store_dir = os.path.join(preds_path, epoch_name)
        os.makedirs(store_dir, exist_ok=True)

        for i, batch in enumerate(val_samples):
            store_preds_path = os.path.join(store_dir, f"exp{i+1}.gif")
            diff_gamma, hyp_diff_gamma, grad_norm_delta = pde_params[i]
            sub_title = (
                f"2D KS auto regressive Predictions with diff_gamma={diff_gamma}, "
                f"hyp_diff_gamma={hyp_diff_gamma}, gradient_norm_delta={grad_norm_delta}"
            )

            traj = torch.tensor(batch, dtype=torch.float32).to(device).squeeze(1)  # shape: (T, H, W)
            time_steps, H, W = traj.shape

            predictions = []
            input_ar = traj[0].unsqueeze(0).unsqueeze(0)  # shape: (1, 1, H, W)
            pde_in = torch.tensor(pde_params[i], dtype=torch.float32).to(device)

            for t in range(time_steps - 1):
                t_tensor = torch.tensor([t], dtype=torch.float32, device=device)
                pde_t = torch.cat((pde_in, t_tensor))
                output = model(input_ar, pde_t)  # shape: (1, 1, H, W)
                output = output.squeeze(0).squeeze(0)     # shape: (H, W)
                predictions.append(output.cpu().numpy())

                input_ar = output.unsqueeze(0).unsqueeze(0).detach()

            targets = traj[1:].cpu().numpy()        # shape: (T-1, H, W)
            predictions = np.stack(predictions)     # shape: (T-1, H, W)

            vmin = min(predictions.min(), targets.min())
            vmax = max(predictions.max(), targets.max())
            data_diff = np.abs(targets - predictions)
            vmin_data, vmax_data = data_diff.min(), data_diff.max()

            fig, axes = plt.subplots(1, 4, figsize=(24, 6))
            fig.suptitle(sub_title, fontsize=16)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])

            # Setup initial images
            im_target = axes[0].imshow(targets[0], cmap='plasma', origin='lower', vmin=vmin, vmax=vmax)
            axes[0].set_title("Targets")
            axes[0].set_xlabel("Width")
            axes[0].set_ylabel("Height")
            plt.colorbar(im_target, ax=axes[0])

            im_pred = axes[1].imshow(predictions[0], cmap='plasma', origin='lower', vmin=vmin, vmax=vmax)
            axes[1].set_title("Auto-regressive Predictions")
            axes[1].set_xlabel("Width")
            axes[1].set_ylabel("Height")
            plt.colorbar(im_pred, ax=axes[1])

            im_diff = axes[2].imshow(data_diff[0], cmap='plasma', origin='lower', vmin=vmin_data, vmax=vmax_data)
            axes[2].set_title("Abs Difference")
            axes[2].set_xlabel("Width")
            axes[2].set_ylabel("Height")
            plt.colorbar(im_diff, ax=axes[2])

            axes[3].plot(np.mean(data_diff, axis=(1,2)), label='Mean Abs Difference')
            axes[3].set_title("Mean Abs Difference Over Time")
            axes[3].set_xlabel("Time Steps")
            axes[3].set_ylabel("Mean Abs Diff")

            def update(frame):
                im_target.set_data(targets[frame])
                axes[0].set_title(f"Targets - Time Step {frame}")

                im_pred.set_data(predictions[frame])
                axes[1].set_title(f"Predictions - Time Step {frame}")

                diff_frame = np.abs(targets[frame] - predictions[frame])
                im_diff.set_data(diff_frame)
                axes[2].set_title(f"Abs Diff - Time Step {frame}")

                return im_target, im_pred, im_diff

            ani = animation.FuncAnimation(
                fig, update, frames=targets.shape[0], interval=200, blit=False
            )

            ani.save(store_preds_path, writer='pillow')
            plt.close()
