import warnings
warnings.filterwarnings('ignore')

import os, random, time
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.model_selection import GroupKFold
from tqdm.notebook import tqdm
from sklearn.model_selection import GroupKFold
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.modules.loss import _WeightedLoss
import torch.nn.functional as F


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def get_data(device):
    print('Loading...')
    data_all = pd.read_parquet('/kaggle/input/janestreet-parquet/train.parquet')
    features = [c for c in data_all.columns if 'feature' in c]
    n_samples = data_all.shape[0]
    train = data_all[:int(0.9 * n_samples)]
    test = data_all[int(0.9 * n_samples):int(0.95 * n_samples)]
    valid = data_all[int(0.95 * n_samples):]

    print('Filling...')
    f_mean = train[features[1:]].mean()
    train = train.loc[train.weight > 0].reset_index(drop = True)
    train[features[1:]] = train[features[1:]].fillna(f_mean)
    train['action'] = (train['resp'] > 0).astype('int')

    #f_mean_test = train[features[1:]].mean()
    test = test.loc[test.weight > 0].reset_index(drop = True)
    test[features[1:]] = test[features[1:]].fillna(f_mean)
    test['action'] = (test['resp'] > 0).astype('int')

    print('Converting...')
    # train = train.to_pandas()
    f_mean = f_mean.values#.get()
    np.save('f_mean.npy', f_mean)

    print('Finish.')
    return train, valid, test, features

class Model(nn.Module):
    def __init__(self, features):
        super(Model, self).__init__()
        self.batch_norm0 = nn.BatchNorm1d(len(features))
        self.dropout0 = nn.Dropout(0.10143786981358652)

        hidden_size = 256
        self.dense1 = nn.Linear(len(features), 384)
        self.batch_norm1 = nn.BatchNorm1d(384)
        self.dropout1 = nn.Dropout(0.19720339053599725)

        self.dense2 = nn.Linear(384, 896)
        self.batch_norm2 = nn.BatchNorm1d(896)
        self.dropout2 = nn.Dropout(0.2703017847244654)

        self.dense3 = nn.Linear(896, 896)
        self.batch_norm3 = nn.BatchNorm1d(896)
        self.dropout3 = nn.Dropout(0.23148340929571917)

        self.dense4 = nn.Linear(896, 394)
        self.batch_norm4 = nn.BatchNorm1d(394)
        self.dropout4 = nn.Dropout(0.2357768967777311)

        self.dense5 = nn.Linear(394, 1)

        self.Relu = nn.ReLU(inplace=True)
        self.PReLU = nn.PReLU()
        self.LeakyReLU = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        # self.GeLU = nn.GELU()
        self.RReLU = nn.RReLU()

    def forward(self, x):
        x = self.batch_norm0(x)
        x = self.dropout0(x)

        x = self.dense1(x)
        x = self.batch_norm1(x)
        x = x * F.sigmoid(x)
        x = self.dropout1(x)

        x = self.dense2(x)
        x = self.batch_norm2(x)
        x = x * F.sigmoid(x)
        x = self.dropout2(x)
        
        x = self.dense3(x)
        x = self.batch_norm3(x)
        x = x * F.sigmoid(x)
        x = self.dropout3(x)
        
        x = self.dense4(x)
        x = self.batch_norm4(x)
        x = x * F.sigmoid(x)
        x = self.dropout4(x)

        x = self.dense5(x)

        return x

class MarketDataset:
    def __init__(self, df, features):
        self.features = df[features].values

        self.label = (df['resp'] > 0).astype('int').values.reshape(-1, 1)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return {
            'features': torch.tensor(self.features[idx], dtype=torch.float),
            'label': torch.tensor(self.label[idx], dtype=torch.float)
        }

class SmoothBCEwLogits(_WeightedLoss):
    def __init__(self, weight=None, reduction='mean', smoothing=0.0):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    @staticmethod
    def _smooth(targets:torch.Tensor, n_labels:int, smoothing=0.0):
        assert 0 <= smoothing < 1
        with torch.no_grad():
            targets = targets * (1.0 - smoothing) + 0.5 * smoothing
        return targets

    def forward(self, inputs, targets):
        targets = SmoothBCEwLogits._smooth(targets, inputs.size(-1),
            self.smoothing)
        loss = F.binary_cross_entropy_with_logits(inputs, targets,self.weight)

        if  self.reduction == 'sum':
            loss = loss.sum()
        elif  self.reduction == 'mean':
            loss = loss.mean()

        return loss

class EarlyStopping:
    def __init__(self, patience=7, mode="max", delta=0.):
        self.patience = patience
        self.counter = 0
        self.mode = mode
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        if self.mode == "min":
            self.val_score = np.Inf
        else:
            self.val_score = -np.Inf

    def __call__(self, epoch_score, model, model_path):

        if self.mode == "min":
            score = -1.0 * epoch_score
        else:
            score = np.copy(epoch_score)

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(epoch_score, model, model_path)
        elif score < self.best_score: #  + self.delta
            self.counter += 1
            print('EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            # ema.apply_shadow()
            self.save_checkpoint(epoch_score, model, model_path)
            # ema.restore()
            self.counter = 0

    def save_checkpoint(self, epoch_score, model, model_path):
        if epoch_score not in [-np.inf, np.inf, -np.nan, np.nan]:
            print('Validation score improved ({} --> {}). Saving model!'.format(self.val_score, epoch_score))
            torch.save(model.state_dict(), model_path)
        self.val_score = epoch_score

def train_fn(model, optimizer, scheduler, loss_fn, dataloader, device):
    model.train()
    final_loss = 0

    for data in tqdm(dataloader):
        optimizer.zero_grad()
        features = data['features'].to(device)
        label = data['label'].to(device)
        outputs = model(features)
        loss = loss_fn(outputs, label)
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()

        final_loss += loss.item()

    final_loss /= len(dataloader)

    return final_loss

def inference_fn(model, dataloader, device):
    model.eval()
    preds = []

    for data in dataloader:
        features = data['features'].to(device)

        with torch.no_grad():
            outputs = model(features)

        preds.append(outputs.sigmoid().detach().cpu().numpy())

    preds = np.concatenate(preds).reshape(-1)

    return preds

def utility_score(date, weight, resp, action):
    count_i = len(np.unique(date))    
    Pi = np.bincount(date, weight * resp * action)
    t = np.sum(Pi) / np.sqrt(np.sum(Pi ** 2)) * np.sqrt(250 / count_i)
    u = np.clip(t, 0, 6) * np.sum(Pi)
    return u

def train():
    batch_size = 4096
    label_smoothing = 1e-2
    learning_rate = 1e-3
    train, _1, _2, features = get_data(device)
    start_time = time.time()
    gkf = GroupKFold(n_splits = 5)
    for fold, (tr, te) in enumerate(gkf.split(train['action'].values, train['action'].values, train['date'].values)):
        print("fold " + str(fold))
        print("Create train dataloader")
        train_set = MarketDataset(train.loc[tr])
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
        print("Create valid dataloader")
        valid_set = MarketDataset(train.loc[te])
        valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=4)
    
        torch.cuda.empty_cache()
        device = torch.device("cuda:0")
        model = Model(features)
        model.to(device)
    
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        loss_fn = SmoothBCEwLogits(smoothing=label_smoothing)
    
        ckp_path = f'JSModel_{fold}.pth'
    
        es = EarlyStopping(patience=3, mode="max")
        for epoch in range(10):
            print("start epoch " + str(epoch))
            train_loss = train_fn(model, optimizer, None, loss_fn, train_loader, device)
            valid_pred = inference_fn(model, valid_loader, device)
            auc_score = roc_auc_score((train.loc[te]['resp'] > 0).astype('int').values.reshape(-1, 1), valid_pred)
            logloss_score = log_loss((train.loc[te]['resp'] > 0).astype('int').values.reshape(-1, 1), valid_pred)
            valid_pred = np.where(valid_pred >= 0.5, 1, 0).astype(int)
            u_score = utility_score(date=train.loc[te].date.values, weight=train.loc[te].weight.values, resp=train.loc[te].resp.values, action=valid_pred)

            print(f"FOLD{fold} EPOCH:{epoch:3}, train_loss:{train_loss:.5f}, u_score:{u_score:.5f}, auc:{auc_score:.5f}, logloss:{logloss_score:.5f}, "
                f"time: {(time.time() - start_time) / 60:.2f}min")
        
            es(auc_score, model, model_path=ckp_path)
            if es.early_stop:
                print("Early stopping")
                break

def evaluation(ckp_path):
    if torch.cuda.is_available():
        print('using device: cuda')
        device = torch.device("cuda:0")
    else:
        print('using device: cpu')
        device = torch.device('cpu')
    _1, _2, test, features = get_data(device)
    torch.cuda.empty_cache()
    #device = torch.device("cuda:0")
    model = Model(features)
    model.to(device)
    model.eval()
    
    model.load_state_dict(torch.load(ckp_path))
    
    gkf = GroupKFold(n_splits = 2)

    for fold, (tr, te) in enumerate(gkf.split(test['action'].values, test['action'].values, test['date'].values)):
        test_set = MarketDataset(test.loc[te])
        test_loader = DataLoader(test_set, batch_size=4096, shuffle=True, num_workers=4)
        test_pred = inference_fn(model, test_loader, device)
        test_pred = np.where(test_pred >= 0.5, 1, 0).astype(int)
        u_score = utility_score(date=test.loc[te].date.values, weight=test.loc[te].weight.values, resp=test.loc[te].resp.values, action=test_pred)
        print(u_score)

if __name__ == "__main__":
    seed_everything(seed=42)
    train()
    evaluation('./JSModel_4.pth')