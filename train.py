import os
import argparse
import pickle

import numpy as np
import pandas as pd
import torch

from utils import CLIPImageEncoder, TextEncoder, OCRExtractor, build_and_save_feature_store
from sklearn.model_selection import train_test_split
import lightgbm as lgb

# Simple Fusion MLP
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class FusionRegressor(nn.Module):
    def __init__(self, img_dim, txt_dim, hidden=1024, dropout=0.2):
        super().__init__()
        self.fc1 = nn.Linear(img_dim + txt_dim, hidden)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, hidden//4)
        self.bn2 = nn.BatchNorm1d(hidden//4)
        self.drop2 = nn.Dropout(dropout)
        self.out = nn.Linear(hidden//4, 1)

    def forward(self, img, txt):
        x = torch.cat([img, txt], dim=1)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.drop1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.drop2(x)
        return self.out(x).squeeze(1)

class EmbeddingDataset(Dataset):
    def __init__(self, store):
        rows = [r for r in store if r['price'] is not None]
        self.X_img = np.stack([r['img_emb'] for r in rows])
        self.X_txt = np.stack([r['text_emb'] for r in rows])
        self.y = np.array([r['price'] for r in rows], dtype=np.float32)
        self.y = np.log1p(self.y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.tensor(self.X_img[idx], dtype=torch.float32), torch.tensor(self.X_txt[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)


def train_mlp(store, model_path, epochs=8, batch_size=64, lr=1e-3, device='cuda'):
    ds = EmbeddingDataset(store)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)
    img_dim = ds.X_img.shape[1]
    txt_dim = ds.X_txt.shape[1]

    model = FusionRegressor(img_dim, txt_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    for ep in range(epochs):
        model.train()
        total_loss = 0.0
        for img, txt, y in dl:
            img = img.to(device)
            txt = txt.to(device)
            y = y.to(device)
            opt.zero_grad()
            preds = model(img, txt)
            loss = F.mse_loss(preds, y)
            loss.backward()
            opt.step()
            total_loss += loss.item() * img.size(0)
        print(f"Epoch {ep+1}/{epochs} MSE: {total_loss/len(ds):.6f}")

    torch.save(model.state_dict(), model_path)
    return model


def train_lgb(store, model_path, num_round=1000):
    rows = [r for r in store if r['price'] is not None]
    X = np.stack([np.concatenate([r['img_emb'], r['text_emb']]) for r in rows])
    y = np.array([r['price'] for r in rows], dtype=np.float32)
    y = np.log1p(y)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42)
    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

    params = {
        'objective': 'regression', 'metric': 'rmse', 'learning_rate': 0.03,
        'num_leaves': 128, 'feature_fraction': 0.8, 'bagging_fraction': 0.9, 'verbosity': -1
    }
    callbacks = [lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=50)]
    model = lgb.train(params, dtrain, num_boost_round=num_round, valid_sets=[dtrain, dval], callbacks=callbacks)
    model.save_model(model_path)
    return model


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    df = pd.read_csv(args.train_csv)

    image_encoder = CLIPImageEncoder()
    text_encoder = TextEncoder()
    ocr = OCRExtractor(use_paddle=True)

    print('Building feature store...')
    store = build_and_save_feature_store(df, image_encoder, text_encoder, ocr, os.path.join(args.output_dir, 'train_embeddings.pkl'))

    print('Training LightGBM...')
    lgb_model = train_lgb(store, os.path.join(args.output_dir, 'lgb_model.txt'), num_round=args.lgb_rounds)

    print('Training Fusion MLP...')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mlp = train_mlp(store, os.path.join(args.output_dir, 'fusion_mlp.pth'), epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, device=device)

    print('Training complete. Models saved to', args.output_dir)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--train_csv', default='dataset/train.csv')
    p.add_argument('--output_dir', default='models')
    p.add_argument('--epochs', type=int, default=8)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--lgb_rounds', type=int, default=1000)
    args = p.parse_args()
    main(args)