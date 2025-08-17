import time
import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
'''
TODO:
    data is formated properly but now need to read into best way to annotate the sequence if they crossed or not also might be good to just compoletly
    edit it to only look at annotations
'''



class StreetCrossingDataset(Dataset):
    def __init__(self, clip_dirs, resize_shape=(416, 416), seq_length=8):
        self.clip_dirs = clip_dirs
        self.resize_shape = resize_shape
        self.seq_length = seq_length
        self.data = []
        for clip_dir in clip_dirs:
            images_dir = os.path.join(clip_dir, "images")
            annotations_dir = os.path.join(clip_dir, "annotations")
            image_files = sorted(os.listdir(images_dir))
            annotation_files = sorted(os.listdir(annotations_dir))
            for img_file, ann_file in zip(image_files, annotation_files):
                img_path = os.path.join(images_dir, img_file)
                ann_path = os.path.join(annotations_dir, ann_file)
                self.data.append((img_path, ann_path))

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        # Load images and annotations for the sequence
        img_seq = []
        label_seq = []
        velocity_seq = []
        for i in range(idx, idx + self.seq_length):
            img_path, ann_path = self.data[i]
            img = cv2.imread(img_path)
            ann = np.loadtxt(ann_path)

            # Preprocess image
            img = cv2.resize(img, self.resize_shape)
            img = img / 255.0
            img = (img - 0.5) / 0.5

            # Convert YOLO annotations to binary labels and velocity
            labels = np.zeros((1,))
            velocity = np.zeros((2,))
            for bbox in ann:
                if bbox.size == 0:
                    continue
                if bbox.size >= 4:
                    x, y, w, h = bbox[:4]
                else:
                    continue
                if x >= 0 and x < self.resize_shape[0] and y >= 0 and y < self.resize_shape[1]:
                    labels[0] = 1
                    # Calculate YOLO box dimensions
                    bbox_width = w * self.resize_shape[0]
                    bbox_height = h * self.resize_shape[1]
                    # Calculate velocity
                    if len(bbox) >= 6:
                        velocity[0] = (x + w/2) - bbox[4]
                        velocity[1] = (y + h/2) - bbox[5]
            # Add image, label, and velocity to sequence
            img_tensor = torch.from_numpy(img.transpose((2, 0, 1))).float()
            label_tensor = torch.from_numpy(labels).float()
            velocity_tensor = torch.from_numpy(velocity).float()
            img_seq.append(img_tensor)
            label_seq.append(label_tensor)
            velocity_seq.append(velocity_tensor)

        # Stack images, labels, and velocities into tensors
        img_seq_tensor = torch.stack(img_seq)
        label_seq_tensor = torch.stack(label_seq)
        velocity_seq_tensor = torch.stack(velocity_seq)

        return img_seq_tensor, label_seq_tensor, velocity_seq_tensor, bbox_width, bbox_height

class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vgg16 = models.vgg16(pretrained=True)
        for param in self.vgg16.parameters():
            param.requires_grad = False
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size = x.size(0)
        x = self.vgg16.features(x)
        x = x.view(batch_size, -1, 512)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        out, _ = self.rnn(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        out = self.sigmoid(out)
        return out
        

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Set hyperparameters
input_size = 512 
hidden_size = 128
output_size = 1
num_layers = 2
batch_size = 4
learning_rate = 0.001
num_epochs = 10

# Load dataset
data_dir = "RNN_data"
clip_dirs = [os.path.join(data_dir, clip_dir) for clip_dir in os.listdir(data_dir)]
dataset = StreetCrossingDataset(clip_dirs)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize model and optimizer
model = RNNModel(input_size, hidden_size, output_size, num_layers).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

criterion = nn.BCELoss()
outputs_list = []
model_path = 'train_rnn.pt'
for epoch in range(num_epochs):
    epoch_loss = 0
    start_time = time.time()
    for i, (images, annotations) in enumerate(train_loader):
        # Move data to device
        images = images.to(device)
        annotations = annotations.to(device)

        # Reshape images
        batch_size, seq_length, channels, height, width = images.shape
        images = images.view(batch_size * seq_length, channels, height, width)
        ano_batch, ano_seq_length, ano_channels = annotations.shape
        annotations = annotations.view(batch_size * seq_length, ano_channels)

        # Forward pass
        outputs = model(images)

        # Compute loss
        loss = criterion(outputs, annotations)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

        
        
        # Print training statistics
        if (i+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
    
    
    epoch_time = time.time() - start_time
    prin = (f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Time: {epoch_time:.2f} seconds')
    print(prin)
    outputs_list.append(prin)
    torch.save(model.state_dict(), model_path)
print(outputs_list)