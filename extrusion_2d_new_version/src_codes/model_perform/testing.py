import torch


def testing_loop(model,device, criterion, test_loader, len_test_dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            output = model(inputs)
            output = output.squeeze(1)
            loss = criterion(output, targets)
            val_loss += loss.detach().cpu().item() * inputs.size(0)

    avg_val_loss = val_loss / len_test_dataset
    return avg_val_loss