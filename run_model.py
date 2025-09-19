#!/usr/bin/env python3

import argparse, os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW,Adam
from torch.optim.lr_scheduler import CyclicLR

from ce_stSENet import CE_stSENet   

parser = argparse.ArgumentParser()
parser.add_argument('--npz', default='chbmit_eeg_segments.npz')
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--batch', type=int, default=32)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--target_len', type=int, default=256*120)  # time samples per segment 2 mins
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:", device)

# -------- Dataset --------
def pad_or_truncate(x, target_len):
    c, t = x.shape
    if t == target_len:
        return x
    if t < target_len:
        return np.pad(x, ((0,0),(0,target_len - t)), mode='constant')
    start = (t - target_len)//2
    return x[:, start:start+target_len]

class SienaDataset(Dataset):
    def __init__(self, npz_file, target_len):
        d1 = np.load('chbmit_eeg_segments.npz', allow_pickle=True)
        d2 = np.load('siena_eeg_segments.npz', allow_pickle=True)

        data1, labels1 = d1['data'], d1['labels']
        data2, labels2 = d2['data'], d2['labels']

        # Make sure they are real Python lists before concatenation
        data  = list(data1) + list(data2)
        labs  = list(labels1) + list(labels2)

        # Now pad or truncate to target length
        self.X = [pad_or_truncate(np.asarray(x), target_len) for x in data]
        self.y = np.array(labs, dtype=np.int64)
        self.target_len = target_len

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        #print(self.X[idx][:30,:].shape)
        arr = torch.tensor(self.X[idx][:23,:], dtype=torch.float32)   # (C,T)
        arr = arr.unsqueeze(1)                                 # (C,1,T)  -> matches model input
        label = torch.tensor(self.y[idx], dtype=torch.long)
        return arr, label

dataset = SienaDataset(args.npz, args.target_len)
n_channels = dataset.X[0].shape[0]
print(f"Loaded {len(dataset)} segments with {n_channels} channels")

# simple 80/20 split
indices = np.arange(len(dataset))
np.random.shuffle(indices)
split = int(0.7 * len(indices))
train_idx, val_idx = indices[:split], indices[split:]
#print(train_idx, val_idx );exit(0)
train_loader = DataLoader(dataset, batch_size=args.batch, sampler=torch.utils.data.SubsetRandomSampler(train_idx))
val_loader   = DataLoader(dataset, batch_size=args.batch, sampler=torch.utils.data.SubsetRandomSampler(val_idx))

# -------- Model --------
model = CE_stSENet(inc=n_channels, class_num=2, si=args.target_len).to(device)
#print(model);exit(0)
criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
#scheduler = CyclicLR(optimizer, base_lr=args.lr/10, max_lr=args.lr,
#                     step_size_up=len(train_loader), cycle_momentum=False)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=args.lr,
    steps_per_epoch=len(train_loader),
    epochs=args.epochs
)
# -------- Training loop --------
def evaluate(loader):
    model.eval()
    loss_sum, correct, total = 0, 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            out, *_ = model(xb, if_pooling=1)   # model returns tuple; first is logits
            loss = criterion(out, yb)
            loss_sum += loss.item() * xb.size(0)
            pred = out.argmax(1)
            correct += (pred == yb).sum().item()
            total += xb.size(0)
    return loss_sum/total, correct/total

for epoch in range(1, args.epochs+1):
    model.train()
    running_loss = 0
    for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
        xb, yb = xb.to(device), yb.to(device)
        #print(xb.shape)
        optimizer.zero_grad()
        out, *_ = model(xb, if_pooling=1)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        scheduler.step()
        running_loss += loss.item() * xb.size(0)
    train_loss = running_loss / len(train_idx)
    val_loss, val_acc = evaluate(val_loader)
    print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

torch.save(model.state_dict(), "ce_stsenet_siena_final.pt")
print("Training complete. Model saved to ce_stsenet_siena_final.pt")
