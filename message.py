from scipy.signal import resample
import neurokit2 as nk
import numpy as np
import torch
from glob import glob
import torch.nn as nn
import os
from collections import OrderedDict
import matplotlib.pyplot as plt

def to_img(ecg,name):
    # ecg = nk.ecg_simulate(duration=15, sampling_rate=1000, heart_rate=80)
    if not os.path.isfile(name):
        signals,info = nk.ecg_process(ecg, sampling_rate=500)
        nk.ecg_plot(signals,info)
        # dir=f"home/sammy/myproject/images/{name[:-4]}.png"
        fig = plt.gcf()
        # fig.set_size_inches(16, 9, forward=True)
        fig.set_size_inches(19, 6, forward=True)
        fig.savefig(name)
class ECGNet(nn.Module):
  def __init__(self):
    super(ECGNet, self).__init__()
    #layer1
    self.layer1_conv2d = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1, 25), stride=(1, 2), bias=True)


    #layer2
    self.layer2_conv2d = nn.Sequential(OrderedDict([
        ("bn1", nn.BatchNorm2d(num_features=32)),
        ("act1", nn.ReLU()),
        ("cn1", nn.Conv2d(32, 64, kernel_size=(1, 15), stride=(1, 1), bias=True)),
        ("bn2", nn.BatchNorm2d(num_features=64)),
        ("act2", nn.ReLU()),
        ("cn2", nn.Conv2d(64, 64, kernel_size=(1, 15), stride=(1, 2),  bias=True)),
        ("bn3", nn.BatchNorm2d(num_features=64)),
        ("act3", nn.ReLU()),
        ("cn3", nn.Conv2d(64, 32, kernel_size=(1, 15), stride=(1, 1), bias=True)),
    ]))
    self.layer2_seModule = nn.Sequential(OrderedDict([
        ("fc1", nn.Conv2d(32, 16, kernel_size=1, bias=True)),
        ("act", nn.ReLU()),
        ("fc2", nn.Conv2d(16, 32, kernel_size=1, bias=True)),
        ("gate", nn.Sigmoid())
    ]))

    #layer3
    self.layer3_conv2d_block1 = nn.Sequential(OrderedDict([
        ("bn1", nn.BatchNorm2d(num_features=32)),
        ("act1", nn.ReLU()),
        ("cn1", nn.Conv2d(32, 64, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0), bias=True)),
        ("bn2", nn.BatchNorm2d(num_features=64)),
        ("act2", nn.ReLU()),
        ("cn2", nn.Conv2d(64, 64, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0), bias=True)),
        ("bn3", nn.BatchNorm2d(num_features=64)),
        ("act3", nn.ReLU()),
        ("cn3", nn.Conv2d(64, 32, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0), bias=True)),
    ]))
    self.layer3_seModule_block1 = nn.Sequential(OrderedDict([
        ("fc1", nn.Conv2d(32, 16, kernel_size=1, bias=True)),
        ("act", nn.ReLU()),
        ("fc2", nn.Conv2d(16, 32, kernel_size=1, bias=True)),
        ("gate", nn.Sigmoid())
    ]))

    self.layer3_conv2d_block2 = nn.Sequential(OrderedDict([
        ("bn1", nn.BatchNorm2d(num_features=32)),
        ("act1", nn.ReLU()),
        ("cn1", nn.Conv2d(32, 64, kernel_size=(5, 1), padding=(2, 0), bias=True)),
        ("bn2", nn.BatchNorm2d(num_features=64)),
        ("act2", nn.ReLU()),
        ("cn2", nn.Conv2d(64, 64, kernel_size=(5, 1), padding=(2, 0), bias=True)),
        ("bn3", nn.BatchNorm2d(num_features=64)),
        ("act3", nn.ReLU()),
        ("cn3", nn.Conv2d(64, 32, kernel_size=(5, 1), padding=(2, 0), bias=True)),
    ]))
    self.layer3_seModule_block2 = nn.Sequential(OrderedDict([
        ("fc1", nn.Conv2d(32, 16, kernel_size=1, bias=True)),
        ("act", nn.ReLU()),
        ("fc2", nn.Conv2d(16, 32, kernel_size=1, bias=True)),
        ("gate", nn.Sigmoid())
    ]))

    self.layer3_conv2d_block3 = nn.Sequential(OrderedDict([
        ("bn1", nn.BatchNorm2d(num_features=32)),
        ("act1", nn.ReLU()),
        ("cn1", nn.Conv2d(32, 64, kernel_size=(7, 1), padding=(3, 0), bias=True)),
        ("bn2", nn.BatchNorm2d(num_features=64)),
        ("act2", nn.ReLU()),
        ("cn2", nn.Conv2d(64, 64, kernel_size=(7, 1), padding=(3, 0), bias=True)),
        ("bn3", nn.BatchNorm2d(num_features=64)),
        ("act3", nn.ReLU()),
        ("cn3", nn.Conv2d(64, 32, kernel_size=(7, 1), padding=(3, 0), bias=True)),
    ]))
    self.layer3_seModule_block3 = nn.Sequential(OrderedDict([
        ("fc1", nn.Conv2d(32, 16, kernel_size=1, bias=True)),
        ("act", nn.ReLU()),
        ("fc2", nn.Conv2d(16, 32, kernel_size=1, bias=True)),
        ("gate", nn.Sigmoid())
    ]))

    #layer4
    self.layer4_conv1d_short_block1 = nn.Sequential(OrderedDict([
        ("bn1", nn.BatchNorm1d(num_features=384)),
        ("act1", nn.ReLU()),
        ("cn1", nn.Conv1d(384, 384, kernel_size=3, stride=9, bias=True)),
    ]))


    self.layer4_conv1d_block1 = nn.Sequential(OrderedDict([
        ("bn1", nn.BatchNorm1d(num_features=384)),
        ("act1", nn.ReLU()),
        ("cn1", nn.Conv1d(384, 768, kernel_size=3, stride=2, bias=True)),
        ("bn2", nn.BatchNorm1d(num_features=768)),
        ("act2", nn.ReLU()),
        ("cn2", nn.Conv1d(768, 768, kernel_size=3, stride=1, bias=True)),
        ("bn3", nn.BatchNorm1d(num_features=768)),
        ("act3", nn.ReLU()),
        ("cn3", nn.Conv1d(768, 1536, kernel_size=3, stride=2, bias=True)),
        ("bn4", nn.BatchNorm1d(num_features=1536)),
        ("act4", nn.ReLU()),
        ("cn4", nn.Conv1d(1536, 384, kernel_size=3, stride=2, bias=True)),
    ]))
    self.layer4_seModule_block1 = nn.Sequential(OrderedDict([
        ("fc1", nn.Conv1d(384, 48, kernel_size=1, bias=True)),
        ("act", nn.ReLU()),
        ("fc2", nn.Conv1d(48, 384, kernel_size=1, bias=True)),
        ("gate", nn.Sigmoid())
    ]))

    self.layer4_conv1d_short_block2 = nn.Sequential(OrderedDict([
        ("bn1", nn.BatchNorm1d(num_features=384)),
        ("act1", nn.ReLU()),
        ("cn1", nn.Conv1d(384, 384, kernel_size=5, stride=9, bias=True)),
    ]))

    self.layer4_conv1d_block2 = nn.Sequential(OrderedDict([
        ("bn1", nn.BatchNorm1d(num_features=384)),
        ("act1", nn.ReLU()),
        ("cn1", nn.Conv1d(384, 768, kernel_size=5, stride=2, padding=2, bias=True)),
        ("bn2", nn.BatchNorm1d(num_features=768)),
        ("act2", nn.ReLU()),
        ("cn2", nn.Conv1d(768, 768, kernel_size=5, stride=2, padding=1, bias=True)),
        ("bn3", nn.BatchNorm1d(num_features=768)),
        ("act3", nn.ReLU()),
        ("cn3", nn.Conv1d(768, 1536, kernel_size=5, stride=1, padding=2, bias=True)),
        ("bn4", nn.BatchNorm1d(num_features=1536)),
        ("act4", nn.ReLU()),
        ("cn4", nn.Conv1d(1536, 384, kernel_size=5, stride=2, padding=1, bias=True)),
    ]))
    self.layer4_seModule_block2 = nn.Sequential(OrderedDict([
        ("fc1", nn.Conv1d(384, 48, kernel_size=1, bias=True)),
        ("act", nn.ReLU()),
        ("fc2", nn.Conv1d(48, 384, kernel_size=1, bias=True)),
        ("gate", nn.Sigmoid())
    ]))

    self.layer4_conv1d_short_block3 = nn.Sequential(OrderedDict([
        ("bn1", nn.BatchNorm1d(num_features=384)),
        ("act1", nn.ReLU()),
        ("cn1", nn.Conv1d(384, 384, kernel_size=7, stride=9, bias=True)),
    ]))

    self.layer4_conv1d_block3 = nn.Sequential(OrderedDict([
        ("bn1", nn.BatchNorm1d(num_features=384)),
        ("act1", nn.ReLU()),
        ("cn1", nn.Conv1d(384, 768, kernel_size=7, stride=2, padding=2, bias=True)),
        ("bn2", nn.BatchNorm1d(num_features=768)),
        ("act2", nn.ReLU()),
        ("cn2", nn.Conv1d(768, 768, kernel_size=7, stride=2, padding=1, bias=True)),
        ("bn3", nn.BatchNorm1d(num_features=768)),
        ("act3", nn.ReLU()),
        ("cn3", nn.Conv1d(768, 1536, kernel_size=7, stride=1, padding=3, bias=True)),
        ("bn4", nn.BatchNorm1d(num_features=1536)),
        ("act4", nn.ReLU()),
        ("cn4", nn.Conv1d(1536, 384, kernel_size=7, stride=2, padding=2, bias=True)),
    ]))
    self.layer4_seModule_block3 = nn.Sequential(OrderedDict([
        ("fc1", nn.Conv1d(384, 48, kernel_size=1, bias=True)),
        ("act", nn.ReLU()),
        ("fc2", nn.Conv1d(48, 384, kernel_size=1, bias=True)),
        ("gate", nn.Sigmoid())
    ]))

    self.layer5_avg_pool1 = nn.AvgPool1d(kernel_size=10)
    self.layer5_avg_pool2 = nn.AvgPool1d(kernel_size=10)
    self.layer5_avg_pool3 = nn.AvgPool1d(kernel_size=10)

    self.fc = nn.Sequential(OrderedDict([
        ("ln1", nn.Linear(1152, 288)),
        ("dp", nn.Dropout(p=0.2)),
        ("act", nn.ReLU()),
        ("ln2", nn.Linear(288, 1)),
        ("sigmoid", nn.Sigmoid())
    ]))

  def forward(self, x):
    #layer1
    x = self.layer1_conv2d(x)

    #layer2
    x = self.layer2_conv2d(x)
    u = x
    x = x.view(x.size(0), x.size(1), -1).mean(-1).view(x.size(0), x.size(1), 1, 1)
    x = self.layer2_seModule(x)
    x = u * x


    #layer3
    x1 = self.layer3_conv2d_block1(x)
    u1 = x1
    x1 = x1.view(x1.size(0), x1.size(1), -1).mean(-1).view(x1.size(0), x1.size(1), 1, 1)
    x1 = self.layer3_seModule_block1(x1)
    x1 = u1 * x1

    x2 = self.layer3_conv2d_block2(x)
    u2 = x2
    x2 = x2.view(x2.size(0), x2.size(1), -1).mean(-1).view(x2.size(0), x2.size(1), 1, 1)
    x2 = self.layer3_seModule_block2(x2)
    x2 = u2 * x2

    x3 = self.layer3_conv2d_block3(x)
    u3 = x3
    x3 = x3.view(x3.size(0), x3.size(1), -1).mean(-1).view(x3.size(0), x3.size(1), 1, 1)
    x3 = self.layer3_seModule_block3(x3)
    x3 = u3 * x3

    #layer4
    x1 = torch.flatten(x1, start_dim=1, end_dim=2)
    x2 = torch.flatten(x2, start_dim=1, end_dim=2)
    x3 = torch.flatten(x3, start_dim=1, end_dim=2)

    # x1 = x1.unsqueeze(1)
    # x2 = x2.unsqueeze(1)
    # x3 = x3.unsqueeze(1)

    x1_short = self.layer4_conv1d_short_block1(x1)

    x1 = self.layer4_conv1d_block1(x1)
    u1 = x1
    x1 = x1.view(x1.size(0), x1.size(1), -1).mean(-1).view(x1.size(0), x1.size(1), 1, 1).flatten(2, 3)
    x1 = self.layer4_seModule_block1(x1)
    x1 = u1 * x1
    x1 = x1 + x1_short

    x2_short = self.layer4_conv1d_short_block2(x2)

    x2 = self.layer4_conv1d_block2(x2)
    u2 = x2
    x2 = x2.view(x2.size(0), x2.size(1), -1).mean(-1).view(x2.size(0), x2.size(1), 1, 1).flatten(2, 3)
    x2 = self.layer4_seModule_block2(x2)
    x2 = u2 * x2
    x2 = x2 + x2_short

    x3_short = self.layer4_conv1d_short_block3(x3)

    x3 = self.layer4_conv1d_block3(x3)
    u3 = x3
    x3 = x3.view(x3.size(0), x3.size(1), -1).mean(-1).view(x3.size(0), x3.size(1), 1, 1).flatten(2, 3)
    x3 = self.layer4_seModule_block3(x3)
    x3 = u3 * x3
    x3 = x3 + x3_short

    x1 = self.layer5_avg_pool1(x1)
    x2 = self.layer5_avg_pool2(x2)
    x3 = self.layer5_avg_pool3(x3)

    x = torch.cat((x1, x2, x3), dim=1).flatten(1)

    x = self.fc(x)

    return x

def load_models(root_local_path, model_class):
  models = []
  paths_local = glob(f"{root_local_path}*")
  for path in paths_local:
    cur_path = glob(f"{path}/*")[0]
    cur_model = model_class()
    cur_model.load_state_dict(torch.load(cur_path,map_location="cpu"))
    models.append({"ill_name" : cur_path.split(path)[1][1:-10].upper(), "model" : cur_model})
  return models

def get_predictions(data, models):

    device = torch.device("cpu")

    label2id = {'NORM': 0,
      'IMI': 1,
      'NDT': 2,
      'NST': 3,
      'LVH': 4,
      'LAFB': 5,
      'IRBBB': 6,
      'IVCD': 7,
      'ASMI': 8,
      'AMI': 9,
      'ISCAL': 10,
      '1AVB': 11,
      'ILMI': 12,
      'ISC': 13,
      'CRBBB': 14,
      'CLBBB': 15,
      'LAO_LAE': 16}

    labels_keys = list(label2id.keys())

    data = np.apply_along_axis(lambda x: nk.ecg_clean(x, sampling_rate=500), axis=1, arr=data)
    peaks = [nk.ecg_findpeaks(data[i])['ECG_R_Peaks'] for i in range(12)]
    peaks_count = [(i, len(i)) for i in peaks]
    peaks = max(peaks_count, key=lambda x: x[1])[0]
    signals = []
    for count, i in enumerate(peaks):
        if count == 0:
            diff2 = abs(peaks[count + 1] - i)
            x = 0
            y = peaks[count + 1] - diff2 // 2
        elif count == len(peaks)-1:
            diff1 = abs(peaks[count - 1] - i)
            x = peaks[count - 1] + diff1 // 2
            y = 5000
        else:
            diff1 = abs(peaks[count - 1] - i)
            diff2 = abs(peaks[count + 1]- i)
            x = peaks[count - 1] + diff1 // 2
            y = peaks[count + 1] - diff2 // 2

        signal = torch.tensor(resample(data[:, x:y], 500, axis=1)).to(device)
        signals.append(signal)
    signals = torch.stack(signals)[:, None, :, :]

    preds = [0] * 17

    signals = signals.to(device)
    for model_item in models:
        model = model_item["model"].to(device)
        model.eval()
        with torch.no_grad():
            output = torch.mean(model(signals.float()).view(1, -1)[0])
            output = float(output.item())
            preds[label2id[model_item["ill_name"]]] = "{:.2f}".format(output)

    return {labels_keys[i] : preds[i] for i in range(17)}
