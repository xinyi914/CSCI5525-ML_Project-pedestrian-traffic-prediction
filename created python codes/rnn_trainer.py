import time
import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from torch.autograd import Variable
import matplotlib as plt
import random



class StreetCrossingDataset(Dataset):
    def __init__(self, clip_dirs, resize_shape=(416, 416), seq_length=16):
        self.clip_dirs = clip_dirs
        self.resize_shape = resize_shape
        self.seq_length = seq_length
        self.data = []

        for clip_dir in clip_dirs:
            if 'clip' in clip_dir:
                annotations_dir = os.path.join(clip_dir, "annotations")
                annotation_files = sorted(os.listdir(annotations_dir))
                if len(annotation_files) < seq_length:
                    continue
                else:
                    self.data.append(clip_dir)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        # Load images and annotations for the sequence
        x_seq = []
        y_seq = []
        velocity_seq = []

        # Get the annotation and output file in idxth clip folder
        clip_dir = self.data[idx]
        label_dir = os.path.join(clip_dir, "output")
        label_file = os.listdir(label_dir)
        annotations_dir = os.path.join(clip_dir, "annotations")
        annotation_files = sorted(os.listdir(annotations_dir))
        # Get the label of this pedestrian from corresponding label path
        lab_path = os.path.join(label_dir, label_file[0])

        labels = np.loadtxt(lab_path)

        # Get the velocity, x, and y from corresponding seq_length frames in annotation path
        for i in range(self.seq_length):

            ann_file = annotation_files[i]
            ann_path = os.path.join(annotations_dir, ann_file)
            ann = np.loadtxt(ann_path)  # the data in one frame
            x, y = ann[1:3]  # get the x, y from the data

            # Calculate velocity
            velocity = 0
            x_prev = 0
            y_prev = 0
            if i == 0:  # if it's the first frame, there's no velocity for it
                x_prev = x
                y_prev = y

            else:
                velocity = np.sqrt((x - x_prev) ** 2 + (y - y_prev) ** 2)  # the velocity for the ith frame
                x_prev = x
                y_prev = y

                # Add velocity to sequence
            velocity_tensor = torch.tensor(velocity).float()
            velocity_seq.append(velocity_tensor)

            # Add x, y to sequence
            x_tensor = torch.tensor(x).float()
            y_tensor = torch.tensor(y).float()
            x_seq.append(x_tensor)
            y_seq.append(y_tensor)

        label_tensor = torch.from_numpy(labels).float()
        x_seq_tensor = torch.stack(x_seq)
        y_seq_tensor = torch.stack(y_seq)
        velocity_seq_tensor = torch.stack(velocity_seq)

        features = torch.stack((x_seq_tensor, y_seq_tensor, velocity_seq_tensor), -1)

        return features, label_tensor


class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, hiddensize2, output_size, num_layers):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True).requires_grad_(True)

        self.rnn2 = nn.LSTM(hidden_size, hiddensize2, num_layers, batch_first=True).requires_grad_(True)

        self.fc = nn.Linear(hiddensize2, output_size)

        self.sigmoid = nn.Sigmoid()



    def forward(self, x):
        batch_size = x.size(0)
        
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).requires_grad_(True)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).requires_grad_(True)
        
        out, hidden = self.rnn(x, (h0,c0))
        #print("First: ", out.requires_grad)
        #out = self.drop(out)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).requires_grad_(True)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).requires_grad_(True)

        out, hidden = self.rnn2(out, (h0,c0))

        out = self.fc(out[:, -1, :])
        
        out = self.sigmoid(out)
        return out
    
    

def predict(model, test_loader):
    tot_accuracy = 0
    for epoch in range(num_epochs):
        epoch_loss = 0
        start_time = time.time()
        accuracy = []
        
        model.eval()
        for i, (feature, label) in enumerate(test_loader):
            outputs = model(feature)
            outputs = outputs.squeeze(1)
            outputs = (outputs > 0.5).float()
            acc = (100 * np.sum((outputs.detach() == label).cpu().numpy()))/label.shape[0]
            accuracy.append(acc)
        tot_accuracy += np.mean(accuracy)
    print("Total Accuracy: ", tot_accuracy/num_epochs)
       
# Set device
device = torch.device("cpu")

# Set hyperparameters
input_size = 3
hidden_size = 200 
output_size = 1
num_layers = 2
batch_size = 4
learning_rate = 0.0001
num_epochs = 10
hidden_size2 = 200
#SGD 0.1, Adam 0.0001
# Load dataset
data_dir = "E:\\data\\Test_RNN_data"
clip_dirs = [os.path.join(data_dir, clip_dir) for clip_dir in os.listdir(data_dir)]
dataset = StreetCrossingDataset(clip_dirs)
train_len = int(len(dataset)*0.8)      
train_set, test_set = torch.utils.data.random_split(dataset, [train_len, len(dataset)-train_len])
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

# Initialize model and optimizer
model = RNNModel(input_size, hidden_size, hidden_size2, output_size, num_layers).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

criterion = nn.BCELoss().to(device)
outputs_list = []
model_path = 'train_rnn2.pt'
for epoch in range(num_epochs):
    epoch_loss = 0
    start_time = time.time()
    model.train()
    for i, (feature, label) in enumerate(train_loader):
        outputs = model(feature)
        outputs = outputs.squeeze(1)
        
        
        loss = criterion(outputs, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        
        epoch_loss += loss.item()   
        acc = 100 * (outputs.detach() == label).cpu().numpy().mean()
        if (i + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Batch [{i + 1}/{train_len}], Loss: {loss.item():.4f}')

    epoch_time = time.time() - start_time
    msg = (f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Time: {epoch_time:.2f} seconds')
    print(msg)
    outputs_list.append(msg)
    torch.save(model.state_dict(), model_path)
    
model = RNNModel(input_size, hidden_size, hidden_size2, output_size, num_layers).to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

print("Test Start")
predict(model, test_loader)

    