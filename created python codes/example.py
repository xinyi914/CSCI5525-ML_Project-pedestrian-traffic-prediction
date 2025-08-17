'''
This just views an yolo example
'''

import cv2

# Set the path to the image and YOLO format text file
img_path = 'OG_training_attempt_data\\frame_25987.jpg'
txt_path = 'OG_training_attempt_data\\frame_25987.txt'

# Read the image
img = cv2.imread(img_path)

# Get the image height and width
h, w = img.shape[:2]

# Read the YOLO format text file and parse the bounding box coordinates
with open(txt_path, 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip().split(' ')
        label = line[0]
        x_center, y_center, box_width, box_height = map(float, line[1:])
        
        # Convert YOLO format to pixel values
        x_min = int((x_center - box_width / 2) * w)
        y_min = int((y_center - box_height / 2) * h)
        x_max = int((x_center + box_width / 2) * w)
        y_max = int((y_center + box_height / 2) * h)
        
        # Draw the bounding box on the image
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(img, label, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Display the image with the bounding box
cv2.imshow('Image with Bounding Box', img)
cv2.waitKey(0)
cv2.destroyAllWindows()