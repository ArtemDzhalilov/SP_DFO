import os
import mne
import numpy as np
import re
import torch
from torch import nn
from torch.nn import functional as F
array_of_useful_channels = ['Fp1-M2', 'C3-M2', 'O1-M2', 'Fp2-M1', 'C4-M1', 'O2-M1']
from catboost import CatBoostRegressor
class Conv_block(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(in_channels, 32, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv1d(in_channels, 32, kernel_size=8, stride=2, padding=3)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x = torch.cat([x1, x2, x3], dim=1)
        x = self.relu(x)
        return x

class Chrononet(nn.Module):
    def __init__(self, in_channels=6):
        super().__init__()
        self.conv_block1 = Conv_block(6)
        self.conv_block2 = Conv_block(96)
        self.conv_block3 = Conv_block(96)
        self.gru1 = nn.GRU(96, 32, batch_first=True)
        self.gru2 = nn.GRU(32, 32, batch_first=True)
        self.gru3 = nn.GRU(64, 32, batch_first=True)
        self.linear = nn.Linear(1875*2, 1)
        self.gru4 = nn.GRU(96, 32, batch_first=True)
        self.last_lin = nn.Linear(32, 2)

    def forward(self, x, return_emb=False):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = x.permute(0, 2, 1)
        x1, _ = self.gru1(x)
        x2, _ = self.gru2(x1)
        x = torch.cat([x1, x2], dim=2)
        x3, _ = self.gru3(x)
        x = torch.cat([x1, x2, x3], dim=2)
        x4 = x.permute(0, 2, 1)
        x = self.linear(x4)
        x = F.relu(x)
        x = x.permute(0, 2, 1)
        x4, _ = self.gru4(x)
        x4 = x4.flatten(1, 2)
        out = self.last_lin(x4)
        if return_emb:
            return out, x4
        else:
            return out
model = Chrononet()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(torch.load('best_model.pth'))
model.eval()
model.to(device)
cat = CatBoostRegressor().load_model('cb_model.cbm')
def read_data_for_inference(datax):
    datax.set_eeg_reference()
    datax.filter(l_freq=1,h_freq=45)
    channels = datax.info.ch_names
    pick_array = []
    skip_channels = []
    for channel in range(len(array_of_useful_channels)):
        if channels[channel] == array_of_useful_channels[channel]:
            pick_array.append(channel)
        else:
            skip_channels.append(channel)
    epochs=mne.make_fixed_length_epochs(datax,duration=150 ,overlap=0)
    epochs=epochs.get_data(picks=pick_array)
    n_inserted = 0
    for s in skip_channels:
        new_channel = np.zeros((1, 30000))
        epochs = np.insert(epochs, s+n_inserted, new_channel, axis=1)
        n_inserted += 1
    return epochs.astype(np.float32)

def predict(data):
    data = torch.from_numpy(data).to(device)
    out, emb = model(data, return_emb=True)
    emb = emb.cpu().detach().numpy().squeeze()
    classes = torch.argmax(out, dim=1)
    arr = out.cpu().detach().numpy()
    max_values = np.max(arr, axis=1)
    top_indices = np.argsort(max_values)[-1:]
    return classes, emb, top_indices

def predict_catboost(emb):
    return cat.predict(emb).mean()

def get_most_popular_by_index(data, inds):
    return data[inds]


def pipeline(datax):
    data = read_data_for_inference(datax)
    out, emb, inds = predict(data)
    out = out.cpu().detach().numpy().tolist()
    if out.count(1) >= out.count(0):
        pred = predict_catboost(emb)
        return 1, data[inds], pred
    else:
        return 0, data[inds], None


