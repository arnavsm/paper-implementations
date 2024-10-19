import torch
from tqdm.auto import tqdm
from typing import Dict, List, Tuple
from utils import save_model
from pathlib import Path

def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float, float]:
    
    """Trains a PyTorch model for a single epoch."""
    
    model.train()
    train_loss, train_acc = 0, 0

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)

    train_loss /= len(dataloader)
    train_acc /= len(dataloader)
    return train_loss, train_acc

def test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module,
              device: torch.device) -> Tuple[float, float]:
    
    """Tests a PyTorch model for a single epoch."""

    model.eval()
    test_loss, test_acc = 0, 0

    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            test_pred_logits = model(X)
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))

    test_loss /= len(dataloader)
    test_acc /= len(dataloader)
    return test_loss, test_acc

def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device,
          checkpoint_dir: str) -> Dict[str, List]:
    
    """Trains and tests a PyTorch model."""

    results = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    model.to(device)
    
    log_dir = Path("vision-transformer/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "training_log.txt"

    with open(log_file, "w") as f:
        f.write("Epoch\tTrain Loss\tTrain Accuracy\tTest Loss\tTest Accuracy\n")

        for epoch in tqdm(range(epochs)):
            train_loss, train_acc = train_step(model=model,
                                              dataloader=train_dataloader,
                                              loss_fn=loss_fn,
                                              optimizer=optimizer,
                                              device=device)
            test_loss, test_acc = test_step(model=model,
                                            dataloader=test_dataloader,
                                            loss_fn=loss_fn,
                                            device=device)
            
            # Print and log results
            log_message = (f"Epoch: {epoch+1} | "
                           f"train_loss: {train_loss:.4f} | "
                           f"train_acc: {train_acc:.4f} | "
                           f"test_loss: {test_loss:.4f} | "
                           f"test_acc: {test_acc:.4f}")
            print(log_message)

            with open(log_file, "a") as f:
                f.write(f"{epoch+1}\t{train_loss:.4f}\t{train_acc:.4f}\t{test_loss:.4f}\t{test_acc:.4f}\n")
            
            results["train_loss"].append(train_loss)
            results["train_acc"].append(train_acc)
            results["test_loss"].append(test_loss)
            results["test_acc"].append(test_acc)

            # Save model checkpoint
            if (epoch + 1) % 5 == 0:  # Save checkpoint every 5 epochs
                checkpoint_path = Path(checkpoint_dir) / f"checkpoint_epoch_{epoch+1}.pt"
                save_model(model, checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pt")

    return results

