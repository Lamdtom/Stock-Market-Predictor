import torch
from torch.utils.data import DataLoader

def evaluate_model(model, dataset, batch_size=32, device=None, criterion=None):
    """
    Evaluate a trained PyTorch model on a dataset.
    
    Args:
        model: PyTorch model (already loaded with weights)
        dataset: PyTorch Dataset (train/val/test)
        batch_size: DataLoader batch size
        device: 'cuda' or 'cpu'
        criterion: loss function (default: MSELoss)
        
    Returns:
        avg_loss: average loss on the dataset
        predictions: all model predictions (tensor)
        targets: all true targets (tensor)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if criterion is None:
        criterion = torch.nn.MSELoss()

    model = model.to(device)
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            total_loss += loss.item() * xb.size(0)
            all_preds.append(pred.cpu())
            all_targets.append(yb.cpu())

    avg_loss = total_loss / len(dataset)
    predictions = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)

    return avg_loss, predictions, targets
