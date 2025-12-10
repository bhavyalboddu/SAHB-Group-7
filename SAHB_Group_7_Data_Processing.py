import os
import librosa
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import math

# Fix deprecated numpy types
np.complex = complex
np.float = float

# Set paths
root_folder = r"C:\Users\bhavy\Downloads\SAHB_Noise Masking\SAHB_Noise Masking\Dataset"
output_csv_path = r"C:\Users\bhavy\Downloads\SAHB_Noise Masking\SAHB_Noise Masking\output_audio_features.csv"

sampling_rate = 44100
samples_per_frame = 2048
frame_duration = 2048/sampling_rate # 46 ms
samples_per_hop = int(0.25 * samples_per_frame) #512
window_duration = 10
frames_per_window = 860
window_hop_length = 1

def extract_features(file_path):
    audio_features = list()
    y, sr = librosa.load(file_path, sr=None)
    num_secs = int(len(y)//sr)
    beg_sec = 0
    end_sec = window_duration

    def extract_features_data_point(start_in_secs):
        extracted_features_per_timestep = list()
        s = int(start_in_secs * sr)
        e = s + samples_per_frame
        abs_end = s + int(window_duration * sr)
        while e <= abs_end:
            sample = y[s:e]

            features = list()
            # chroma_stft
            features.append(np.mean(librosa.feature.chroma_stft(y=sample, sr=sr)))
            # rmse
            features.append(np.mean(librosa.feature.rms(y=sample)))
            # spectral_centroid
            features.append(np.mean(librosa.feature.spectral_centroid(y=sample, sr=sr)))
            # spectral_bandwidth
            features.append(np.mean(librosa.feature.spectral_bandwidth(y=sample, sr=sr)))
            # rolloff
            features.append(np.mean(librosa.feature.spectral_rolloff(y=sample, sr=sr)))
            # zero_crossing_rate
            features.append(np.mean(librosa.feature.zero_crossing_rate(y=sample)))
            # mel_mean
            features.append(np.mean(librosa.power_to_db(librosa.feature.melspectrogram(y=sample, sr=sr))))
            # mel_std
            features.append(np.std(librosa.power_to_db(librosa.feature.melspectrogram(y=sample, sr=sr))))
            # fft_mean
            features.append(np.mean(np.abs(np.fft.rfft(sample))))
            # fft_std
            features.append(np.std(np.abs(np.fft.rfft(sample))))
            # onset_mean
            features.append(np.mean(librosa.onset.onset_strength(y=sample, sr=sr)))
            # onset_std
            features.append(np.std(librosa.onset.onset_strength(y=sample, sr=sr)))
            # mfcc and mfcc_delta
            mfccs = librosa.feature.mfcc(y=sample, sr=sr, n_mfcc=13)
            delta_mfccs = librosa.feature.delta(mfccs)

            for i in range(13):
                features.append(np.mean(mfccs[i]))
                features.append(np.mean(delta_mfccs[i]))
            
            extracted_features_per_timestep.append(np.array(features))
            
            s += samples_per_hop
            e += samples_per_hop
        
        return np.array(extracted_features_per_timestep)

    while end_sec <= num_secs:
        audio_features.append(extract_features_data_point(beg_sec))
        beg_sec += window_hop_length
        end_sec += window_hop_length

    return np.array(audio_features)

class AudioWindowDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Take last time step
        return out

audio_file_path = r"C:\Users\bhavy\Downloads\SAHB_Noise Masking\SAHB_Noise Masking\Dataset\Alarm\alarm-104243.mp3"
# this part about labels is all dummy stuff because we dont have our actual data yet
# file path definitely has to change and so does the way that we handle this data
labels_file_path = r"C:\Users\bhavy\Downloads\SAHB_Noise Masking\SAHB_Noise Masking\audio-labels.csv"
feature_list = extract_features(audio_file_path)

# padding for uniformity in numpy array sizes

max_frames = max(len(w) for w in feature_list)
num_features = len(feature_list[0][0])

padded_features = []
for w in feature_list:
    if w.shape[0] < max_frames:
        pad = np.zeros((max_frames - w.shape[0], num_features))
        w = np.vstack([w, pad])
    padded_features.append(w)

padded_features = np.array(padded_features)

# will most definitely have to change this based on how you guys label the data

labels = pd.read_csv(labels_file_path)

X = torch.tensor(padded_features, dtype=torch.float32)
y = torch.tensor(labels, dtype=torch.float32)
y = y.view(-1, 1)

# train-validation-test split

num_windows = len(padded_features)
train_frac = 0.7
val_frac = 0.15
test_frac = 0.15

train_end = int(train_frac * num_windows)
val_end = train_end + int(val_frac * num_windows)

X_train = X[:train_end]
y_train = y[:train_end]

X_val = X[train_end:val_end]
y_val = y[train_end:val_end]

X_test = X[val_end:]
y_test = y[val_end:]

train_audio_dataset = AudioWindowDataset(X_train, y_train)
val_audio_dataset = AudioWindowDataset(X_val, y_val)
test_audio_dataset = AudioWindowDataset(X_test, y_test)

batch_size = 32

train_dataloader = DataLoader(train_audio_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_audio_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_audio_dataset, batch_size=batch_size, shuffle=False)

input_dim = X.shape[2]
model = LSTMModel(input_dim=input_dim, hidden_dim=128, layer_dim=2, output_dim=1)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# may need to change the number of epochs based on overfitting
# training and validation of data

n_epochs = 15

for epoch in range(n_epochs):

    model.train()
    train_loss = float()
    train_correct = 0
    train_total = 0

    for X_batch, y_batch in train_dataloader:
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * y_batch.size(0)
        predicted_labels = (torch.sigmoid(y_pred) > 0.5).int()
        train_correct += (predicted_labels == y_batch.int()).sum().item()
        train_total += y_batch.size(0)

    train_loss = train_loss/train_total
    train_accuracy = train_correct/train_total

    # now its time for validation --> used to tweak hyperparameters, etc.
    
    model.eval()
    val_loss = float()
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for X_batch, y_batch in val_dataloader:
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)

            val_loss += loss.item() * y_batch.size(0)
            predicted_labels = (torch.sigmoid(y_pred) > 0.5).int()
            val_correct += (predicted_labels == y_batch.int()).sum().item()
            val_total += y_batch.size(0)
        
    val_loss = val_loss/val_total
    val_accuracy = val_correct/val_total

    print(f"Epoch {epoch+1}/{n_epochs} | " +  f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f} | " +  f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

# testing data

model.eval()
test_loss = float()
test_correct = 0
test_total = 0
all_test_preds = list()
all_test_labels = list()
all_test_probs = list()

with torch.no_grad():
    for X_batch, y_batch in test_dataloader:
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)

        test_loss += loss.item() * y_batch.size(0)
        predicted_labels = (torch.sigmoid(y_pred) > 0.5).int()
        test_correct += (predicted_labels == y_batch.int()).sum().item()
        test_total += y_batch.size(0)
        all_test_preds.extend(predicted_labels)
        all_test_labels.extend(y_batch)
        all_test_probs.extend(y_pred)

test_loss = test_loss/test_total
test_accuracy = test_correct/test_total

print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.4f} | ")

df = pd.DataFrame({
    "True_Label": all_test_labels.numpy().flatten(),
    "Prediction_Label": all_test_preds.numpy().flatten(),
    "Predicted Probabilities": all_test_probs.numpy().flatten(),
    "Smoothed Probabilities": all_test_probs.numpy().flatten(),
    "White Noise Actuator Setting": None
})

# handling the white noise actuator
# using exponential moving average (EMA) probabilities and hysteresis

def ema(curr_prob, prev_ema, hp):
    return (curr_prob*hp) + (prev_ema*(1-hp))

# the following parameters are NOT set in stone and have to be tuned based on our results (do testing to see which combo gives best results with fewest changes in white noise actuator setting)

ema_hyperparamater = 0.3
hysteresis_low_thresh = 0.4
hysteresis_high_thresh = 0.6

for i in range(len(df)):
    if i == 0:
        curr_prob = df.loc[i, "Predicted Probabilities"]
        df.loc[i, "Smoothed Probabilities"] = curr_prob
        if curr_prob < hysteresis_high_thresh:
            df.loc[i, "White Noise Actuator Setting"] = "NO"
        else:
            df.loc[i, "White Noise Actuator Setting"] = "YES"
    else:
        prev_ema = ema(df.loc[i, "Predicted Probabilities"], df.loc[i-1, "Smoothed Probabilities"], ema_hyperparamater)
        df.loc[i, "Smoothed Probabilities"] = prev_ema
        if prev_ema <= hysteresis_low_thresh:
            df.loc[i, "White Noise Actuator Setting"] = "NO"
        elif prev_ema >= hysteresis_high_thresh:
            df.loc[i, "White Noise Actuator Setting"] = "YES"
        else:
            df.loc[i, "White Noise Actuator Setting"] = df.loc[i-1, "White Noise Actuator Setting"]

# note that above, if white noise actuator setting is "YES", the actuator should be on, and if it is "NO", it should be turned off

df.to_csv("white_noise_actuator_results.csv", index=False)

# need to write code for creating figures