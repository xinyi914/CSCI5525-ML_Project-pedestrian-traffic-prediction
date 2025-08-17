
import cv2
import torch
import torch.nn as nn
import numpy as np
import random
from hubconf import custom

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

yolo = custom(path_or_model='Yolo\\yolov7-main\\runs\\train\\exp19\\weights\\best.pt')

#you can fill in this line with any video from the folder test_vids
cap = cv2.VideoCapture("Yolo\\yolov7-main\\test_vids\\4.avi")

classes = ['person', 'null', 'person_walking']

color = (0, 255, 0)

font = cv2.FONT_HERSHEY_PLAIN
full_detection_list = []
##################################
#yolo extraction
##################################
while True:
    if len(full_detection_list) == 16:
        break
    ret, frame = cap.read()
    if not ret:
        break
    results = yolo(frame, size=640)
    detections = []
    for result in results.xyxy[0]:
        if classes[int(result[5])] in ['person', 'null', 'person_walking']:
            detections.append(result)

    for detection in detections:   
        x1, y1, x2, y2, conf, cls = detection
        if conf > .6:
            if cls == 2:
                full_detection_list.append(detection)
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                class_name = classes[int(cls)]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, class_name, (x1, y1 - 5), font, 1, color, 2)

    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

###############################
##process yolo data
################################
detection_tensor = torch.tensor([list(detection[:4]) for detection in full_detection_list])
prev_x = 0
prev_y = 0
velocities = []

for i in range(len(full_detection_list)):
    x1, y1, x2, y2, conf, cls = full_detection_list[i].cpu().numpy()
    velocity = np.sqrt((x1 - prev_x) ** 2 + (y1 - prev_y) ** 2)
    velocity = velocity.astype(np.float32)
    prev_y = y1
    prev_x = x1
    velocities.append(velocity)

# Convert full_detection_list and velocities to numpy arrays
full_detection_array = detection_tensor.cpu().numpy()
velocities_array = np.array(velocities)

# Create tensor with x1, x2, and velocities
detection_tensor = torch.from_numpy(np.stack((full_detection_array[:, 0], full_detection_array[:, 2], velocities_array), axis=1)).unsqueeze(0)

##########################
#RNN prediction
#########################


input_size = 3
hidden_size = 200 
output_size = 1
num_layers = 2
hiddensize2 = 200

model_path = 'train_rnn2.pt'


class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, hiddensize2, output_size, num_layers):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.hidden_size2 = hiddensize2
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True).requires_grad_(True)

        self.rnn2 = nn.LSTM(hidden_size, hiddensize2, num_layers, batch_first=True).requires_grad_(True)

        self.fc = nn.Linear(hiddensize2, output_size)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size = x.size(0)

        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).requires_grad_(True)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).requires_grad_(True)

        out, hidden = self.rnn(x, (h0,c0))
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).requires_grad_(True)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).requires_grad_(True)

        out, hidden = self.rnn2(out, (h0,c0))

        out = self.fc(out[:, -1, :])

        out = self.sigmoid(out)
        return out

model = RNNModel(input_size, hidden_size, hiddensize2, output_size, num_layers)
model.load_state_dict(torch.load(model_path))
model.eval()


outputs = model(detection_tensor)
outputs = outputs.squeeze(1)
outputs = (outputs > 0.5).float()
print(outputs)