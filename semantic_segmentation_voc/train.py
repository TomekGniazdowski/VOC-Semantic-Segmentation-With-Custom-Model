import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import matplotlib.pyplot as plt


def accuracy(y_pred: torch.Tensor, y_true: torch.Tensor):
    preds = torch.argmax(y_pred, dim=1)
    not_bg = (y_true != 255).float()
    return ((preds == y_true).float() * not_bg).sum() / not_bg.sum()

def validate(
    model: nn.Module, 
    loss_fn: nn.Module, 
    dataloader: DataLoader,
    device: str
):
    total_loss = 0
    total_acc = 0
    batch_ctr = 0
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            y_pred = model(X_batch.to(device))
            total_loss += loss_fn(y_pred, y_batch.to(device)).sum()
            total_acc += accuracy(y_pred, y_batch.to(device))
            batch_ctr += 1
    return (total_loss / batch_ctr).item(), (total_acc / batch_ctr).item()

def train(
    model: nn.Module, 
    optimizer: optim.Optimizer, 
    scheduler: optim.lr_scheduler,
    loss_fn: nn.Module, 
    train_dl: DataLoader, 
    val_dl: DataLoader, 
    epochs: int, 
    patience: int,
    l1_reg_coef: float,
    best_model_path: str,
    device: str,
    print_metrics: bool = True
):
    
    pbar = tqdm(range(epochs))
    best_val_acc = 0
    patience_counter = 0
    train_ds_info = {'loss': [], 'acc': []}
    val_ds_info = {'loss': [], 'acc': []}
    
    for epoch in pbar:
        model.train()
        for X_batch, y_batch in train_dl:
            y_pred = model(X_batch.to(device))
            l1_reg_loss = sum([torch.sum(torch.abs(param)) for param in model.parameters()])
            loss = loss_fn(y_pred, y_batch.to(device)) + l1_reg_coef * l1_reg_loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        model.eval()
        train_loss, train_acc = validate(model=model, loss_fn=loss_fn, dataloader=train_dl, device=device)
        val_loss, val_acc = validate(model=model, loss_fn=loss_fn, dataloader=val_dl, device=device)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = model.state_dict()
            torch.save(best_model, best_model_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter > patience:
                print(f"Early stopping at epoch {epoch}.")
                break
        
        train_ds_info['loss'].append(train_loss)
        train_ds_info['acc'].append(train_acc)
        val_ds_info['loss'].append(val_loss)
        val_ds_info['acc'].append(val_acc)
        
        if print_metrics: 
            pbar.set_postfix({
                "Epoch": epoch,
                "train loss": round(train_loss, 3),
                "train acc": round(train_acc, 3),
                "val loss": round(val_loss, 3),
                "val acc": round(val_acc, 3)
            })

    _, axs = plt.subplots(2, 1, figsize=(15, 5))
    axs[0].set_title('Loss')
    axs[0].set_ylabel('Value')
    axs[0].plot(train_ds_info['loss'], label='train loss', marker='o')
    axs[0].plot(val_ds_info['loss'], label='val loss', marker='*')
    axs[0].legend()
    
    axs[1].set_title('Accuracy')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Value')
    axs[1].plot(train_ds_info['acc'], label='train acc', marker='o')
    axs[1].plot(val_ds_info['acc'], label='val acc', marker='*')
    axs[1].legend()
    plt.show()