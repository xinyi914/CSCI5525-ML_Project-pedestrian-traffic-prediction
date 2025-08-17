import os
import random
import shutil

# Path to the folder containing the image and annotation files
data_folder = "OG_training_attempt_data"

# Path to the destination folder where you want to move the train, valid, and test sets
output_folder = "Yolo\yolov7-main\ogdataidea"

# Create the train, valid, and test folders
train_folder = os.path.join(output_folder, "train")
valid_folder = os.path.join(output_folder, "valid")
test_folder = os.path.join(output_folder, "test")
os.makedirs(train_folder, exist_ok=True)
os.makedirs(valid_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

# Set the percentage of images to use for the train, validation, and test sets
train_percent = 0.7
valid_percent = 0.2
test_percent = 0.1

# Get a list of all the image files in the data folder
image_files = [f for f in os.listdir(data_folder) if f.endswith(".jpg")]

# Shuffle the list of image files randomly
random.shuffle(image_files)

# Calculate the number of images to use for each set
num_images = len(image_files)
num_train = int(train_percent * num_images)
num_valid = int(valid_percent * num_images)
num_test = int(test_percent * num_images)

# Loop through the image files and move them to the appropriate set folder
for i, image_file in enumerate(image_files):
    # Determine the set to which this image belongs
    if i < num_train:
        set_folder = train_folder
    elif i < num_train + num_valid:
        set_folder = valid_folder
    else:
        set_folder = test_folder
    
    # Move the image file to the set folder
    src_image_path = os.path.join(data_folder, image_file)
    dst_image_path = os.path.join(set_folder, image_file)
    shutil.copy(src_image_path, dst_image_path)
    
    # Move the annotation file to the set folder
    annotation_file = image_file.replace(".jpg", ".txt")
    src_annotation_path = os.path.join(data_folder, annotation_file)
    dst_annotation_path = os.path.join(set_folder, annotation_file)
    shutil.copy(src_annotation_path, dst_annotation_path)

# Create the YOLOv5 data.yaml file
with open(os.path.join(output_folder, "data.yaml"), "w") as f:
    f.write("train: train\n")
    f.write("val: valid\n")
    f.write("test: test\n")
    f.write("nc: 1\n")
    f.write("names: ['object']\n")
    
# Create the YOLOv5 train.txt, valid.txt, and test.txt files
with open(os.path.join(output_folder, "train.txt"), "w") as f:
    for image_file in os.listdir(train_folder):
        if image_file.endswith(".jpg"):
            f.write(os.path.join(train_folder, image_file) + "\n")
            
with open(os.path.join(output_folder, "valid.txt"), "w") as f:
    for image_file in os.listdir(valid_folder):
        if image_file.endswith(".jpg"):
            f.write(os.path.join(valid_folder, image_file) + "\n")
            
with open(os.path.join(output_folder, "test.txt"), "w") as f:
    for image_file in os.listdir(test_folder):
        if image_file.endswith(".jpg"):
            f.write(os.path.join(test_folder, image_file) + "\n")