import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import load_datasets
from model import LSTMModel  # make sure you have this
from eval import evaluate_model

def train_model(
    csv_path,
    lookback=10,
    preprocessed=False,
    batch_size=32,
    epochs=30,
    lr=1e-3,
    model_path="model/model.pt",
    device=None
):
    """
    Train a stock prediction model.
    
    Args:
        csv_path: path to CSV file (raw or preprocessed)
        lookback: sliding window size
        preprocessed: if True, CSV already has features
        batch_size: DataLoader batch size
        epochs: number of training epochs
        lr: learning rate
        model_path: where to save the best model
        device: 'cuda' or 'cpu', default auto
    Returns:
        best_val_loss: validation loss of best model
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load datasets
    train_ds, val_ds, test_ds = load_datasets(
        csv_path, lookback=lookback, preprocessed=preprocessed
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Model, loss, optimizer
    input_size = train_ds.X.shape[-1]
    model = LSTMModel(input_size=input_size).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float('inf')

    # Training loop
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = criterion(pred, yb)
                val_loss += loss.item() * xb.size(0)
        val_loss /= len(val_loader.dataset)

        print(f"Epoch {epoch}/{epochs} - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path)
            print(f"Saved new best model: {model_path}")

    # Test evaluation
    model.load_state_dict(torch.load(model_path))
    test_loss, preds, targets = evaluate_model(model, test_ds, batch_size=batch_size)
    print(f"Final Test Loss: {test_loss:.6f}")

    return best_val_loss, test_loss

if __name__ == '__main__':
    best_val, test_loss = train_model(
        csv_path="../data/AAPL_raw.csv",
        lookback=10,
        preprocessed=False,
        batch_size=32,
        epochs=20,
        lr=1e-3,
        model_path="model/model.pt"
    )
    print(f"Training complete. Best Val Loss: {best_val:.6f}, Test Loss: {test_loss:.6f}")