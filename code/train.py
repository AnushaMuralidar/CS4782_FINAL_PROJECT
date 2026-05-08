import os
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from dataset import NPZDataset
from losses import baseline_loss, unet_loss
from evaluate import compute_error

from models.baseline import DLocBaseline
from models.unet import DLocUNet
from models.unet_transformer import DLocUNetTransformer

def get_model(model_name):
    if model_name == "baseline":
        return DLocBaseline()
    elif model_name == "unet":
        return DLocUNet()
    elif model_name == "unet_transformer":
        return DLocUNetTransformer()
    else:
        raise ValueError(f"Unknown model: {model_name}")
    

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default="unet",
                        choices=["baseline", "unet", "unet_transformer"])
    parser.add_argument("--data_path", type=str, default="../data")
    parser.add_argument("--save_dir", type=str, default="../results")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)

    args = parser.parse_args()
    
    # DEVICE
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device:", device)

    os.makedirs(args.save_dir, exist_ok=True)

    dataset = NPZDataset(args.data_path)

    train_data, test_data = train_test_split(dataset.samples, test_size=0.3, random_state=42)

    train_dataset = NPZDataset(args.data_path)
    train_dataset.samples = train_data

    test_dataset = NPZDataset(args.data_path)
    test_dataset.samples = test_data

    use_cuda = torch.cuda.is_available()

    train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True, num_workers = 2 if use_cuda else 0, pin_memory = use_cuda)
    test_loader  = DataLoader(test_dataset, batch_size = args.batch_size, shuffle=False, num_workers = 2 if use_cuda else 0, pin_memory = use_cuda)

    model = get_model(args.model).to(device)

    if args.model == "baseline":
        loss_fn = baseline_loss
    else:
        loss_fn = unet_loss

    optimizer = optim.Adam(model.parameters(), lr = args.lr, weight_decay = 1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # TRAIN

    best_median = float('inf')
    metrics_log = []

    for epoch in range(args.epochs):

        start = time.time()

        model.train()
        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")

        for A, B, L in loop:
            A, B, L = A.to(device), B.to(device), L.to(device)

            loc_pred, cons_pred = model(A)
            
            loc_pred  = F.interpolate(loc_pred,  size=(L.shape[2], L.shape[3]))
            cons_pred = F.interpolate(cons_pred, size=(B.shape[2], B.shape[3]))
            
            loss, l_loc, l_cons = loss_fn(loc_pred, cons_pred, L, B)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loop.set_postfix(loss=loss.item(), loc=l_loc, cons=l_cons)

        scheduler.step()
        # EVAL
        model.eval()
        errors = []

        with torch.no_grad():
            for A, B, L in test_loader:
                A, L = A.to(device), L.to(device)
        
                loc_pred, _ = model(A)
                loc_pred = F.interpolate(loc_pred, size=(L.shape[2], L.shape[3]))
        
                for i in range(A.shape[0]):
                    errors.append(compute_error(loc_pred[i:i+1], L[i]))

        errors = np.array(errors)

        median = np.median(errors)
        p90    = np.percentile(errors, 90)
        epoch_time = time.time() - start

        print(f"\nEpoch {epoch+1} DONE")
        print(f"Median: {median*100:.1f} cm | P90: {p90*100:.1f} cm")
        print(f"Time: {epoch_time:.2f}s\n")

        # SAVE BEST MODEL
        if median < best_median:
            best_median = median
            checkpoint_path = os.path.join(args.save_dir, f"best_{args.model}.pt")
            torch.save(model.state_dict(), checkpoint_path)
            print("Saved best model\n")

        metrics_log.append({
            "epoch": epoch+1,
            "median_cm": median*100,
            "p90_cm": p90*100,
            "time_sec": epoch_time
        })

    # SAVE RESULTS
    df = pd.DataFrame(metrics_log)
    df.to_csv(os.path.join(args.save_dir, "training_metrics.csv"), index=False)

    np.save(os.path.join(args.save_dir, "final_errors.npy"), errors)

    print(" Saved all results!")


if __name__ == "__main__":
    main()