import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import multiprocessing
from joblib import Parallel, delayed
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score, precision_recall_curve, fbeta_score
from EarlyStopping import EarlyStopping
from tqdm import tqdm
import Utils as u
num_cores = multiprocessing.cpu_count()
import pickle


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # 1D convolutional network for ECG signal encoding
        # Input: [batch_size, 1, 256]
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, stride=2, padding=2),  # -> [batch_size, 16, 128]
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),  # -> [batch_size, 32, 64]
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),  # -> [batch_size, 64, 32]
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),  # -> [batch_size, 128, 16]
            nn.ReLU()
        )
        
    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        # 1D transposed convolution network for ECG signal decoding
        # Input: [batch_size, 128, 16]
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1),  # -> [batch_size, 64, 32]
            nn.ReLU(),
            nn.ConvTranspose1d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=1),  # -> [batch_size, 32, 64]
            nn.ReLU(),
            nn.ConvTranspose1d(32, 16, kernel_size=5, stride=2, padding=2, output_padding=1),  # -> [batch_size, 16, 128]
            nn.ReLU(),
            nn.ConvTranspose1d(16, 1, kernel_size=5, stride=2, padding=2, output_padding=1),  # -> [batch_size, 1, 256]
            nn.Tanh()  # Output between -1 and 1 (normalized ECG signal)
        )
        
    def forward(self, x):
        return self.decoder(x)


print("ok")