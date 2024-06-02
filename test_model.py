import random

import numpy as np
from pathlib import Path

from model import Chrononet
import torch
import os
import re
import mne


model = Chrononet()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(torch.load('best_model.pth'))
model.eval()
model.to(device)


arr = []

def calculate_accuracy(outputs, labels):
    preds = torch.argmax(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


def calculate_f1_score(outputs, labels):
    preds = torch.argmax(outputs, dim=1)
    tp = torch.sum(preds & labels).item()
    fp = torch.sum(preds & ~labels).item()
    fn = torch.sum(~preds & labels).item()

    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0

    f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

    return f1_score


def calculate_precision(outputs, labels):
    preds = torch.argmax(outputs, dim=1)
    tp = torch.sum(preds & labels).item()
    fp = torch.sum(preds & ~labels).item()

    precision = tp / (tp + fp) if tp + fp > 0 else 0

    return precision



#Dataset with channel masking
class DatasetWithMasks(torch.utils.data.Dataset):
    def __init__(self, data, num_channels=6, zero_out_probability=0.5):
        self.data = []
        for sequence, label in data:
            self.data.append((sequence[:6], label))
        self.num_channels = num_channels
        self.zero_out_probability = zero_out_probability

    def __len__(self):
        return 2 * len(self.data)

    def __getitem__(self, idx):
        sequence, label = self.data[idx // 2]
        sequence_tensor = torch.FloatTensor(sequence)
        label_tensor = torch.tensor(label, dtype=torch.long)

        if idx % 2 == 0:
            return sequence_tensor, label_tensor

        augmented_sequence = sequence_tensor.clone()
        for i in range(min(self.num_channels, len(augmented_sequence))):
            if random.random() < self.zero_out_probability:
                augmented_sequence[i] = torch.zeros_like(augmented_sequence[i])

        return augmented_sequence, label_tensor

#Dataset without channel masking
class DatasetWithoutMasks(torch.utils.data.Dataset):
    def __init__(self, data, num_channels=6, zero_out_probability=0.5):
        self.data = []
        for sequence, label in data:
            self.data.append((sequence[:6], label))
        self.num_channels = num_channels
        self.zero_out_probability = zero_out_probability

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sequence, label = self.data[idx]
        sequence_tensor = torch.FloatTensor(sequence)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return sequence_tensor, label_tensor

def execute(path):

    files = os.listdir(path)
    #print(files)
    paths1 = [os.path.join(path, f) for f in files if f.endswith(".REC") or f.endswith(".rec")]
    if len(paths1)==0:
        return 0
    old_filename = Path(paths1[0])

    new_extension = '.EDF'

    new_filename = old_filename.with_suffix(new_extension)
    try:
        old_filename.rename(new_filename)
    except:
        pass

def read_data(path):
    match = re.search(r'\d+', path)
    files = os.listdir(path)
    files = [os.path.join(path, f) for f in files if f.endswith(".EDF")]
    for file in files:
        datax=mne.io.read_raw_edf(file,preload=True)
        datax.set_eeg_reference()
        datax.filter(l_freq=1,h_freq=45)
        epochs=mne.make_fixed_length_epochs(datax,duration=150 ,overlap=0)
        epochs=epochs.get_data()
        arr.append(match.group())
        if match.group() == '1':
            return [epochs.astype(np.float32), 0]
        else:
            return [epochs.astype(np.float32), 1]
paths = []

if len(paths) == 0:
    raise Exception("Заполните переменную paths своими значниями (используйте глобальные пути и и указыввайте название то Nr файла. Пример - C:/Users/User/Downloads/sample/Np 18/Nr 1/)")
data = []

for s in paths:
    execute(s)
    data.append(read_data(s))



transformed_data2 = []
for item in data:

    try:
        for sequence in item[0]:
            transformed_data2.append((sequence, item[1]))
    except:
        pass


dataset = DatasetWithMasks(transformed_data2, num_channels=6, zero_out_probability=0.5)
dataset2 = DatasetWithoutMasks(transformed_data2, num_channels=6)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
dataloader2 = torch.utils.data.DataLoader(dataset2, batch_size=64, shuffle=True)

from catboost import CatBoostRegressor
arr = []
res = []
with torch.no_grad():
    accuracy = 0
    f1 = 0
    precision = 0
    recall = 0
    for i, data in enumerate(dataloader, 0):
        inputs, labels = data
        labels = labels.to(device)
        inputs = inputs.to(device)
        outputs= model(inputs, return_emb=False)
        accuracy += calculate_accuracy(outputs, labels)
        f1 += calculate_f1_score(outputs, labels)
        precision += calculate_precision(outputs, labels)
    print(f"Accuracy with masks: {accuracy / len(dataloader)}")
    print(f"F1 with masks: {f1 / len(dataloader)}")
    print(f"Precision with masks: {precision / len(dataloader)}")
    print()
    print()
    accuracy = 0
    f1 = 0
    precision = 0
    recall = 0
    for i, data in enumerate(dataloader2, 0):
        inputs, labels = data
        labels = labels.to(device)
        inputs = inputs.to(device)
        outputs= model(inputs, return_emb=False)
        accuracy += calculate_accuracy(outputs, labels)
        f1 += calculate_f1_score(outputs, labels)
        precision += calculate_precision(outputs, labels)
    print(f"Accuracy without masks: {accuracy / len(dataloader2)}")
    print(f"F1 without masks: {f1 / len(dataloader2)}")
    print(f"Precision without masks: {precision / len(dataloader2)}")
