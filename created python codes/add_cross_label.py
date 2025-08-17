import cv2
import os

"""
This code will play each folder and at the end prompt you if they crossed this will then add that label to the data so we can process it
for the rnn sequence style
"""

# Define the directory paths
site_dir = "E:\data\Test_RNN_data"


# Get a list of all image files in the images directory
for dir in os.listdir(site_dir):
    folder_dir = os.path.join(site_dir, dir)
    images_dir = os.path.join(folder_dir, "images")
    annotations_dir = os.path.join(folder_dir, "annotations")
    output_dir = os.path.join(folder_dir, "output")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    

    image_files = [f for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f))]

    # Iterate through each image file
    for image_file in image_files:
        # Load the image
        image_path = os.path.join(images_dir, image_file)
        image = cv2.imread(image_path)
        
        # Load the corresponding annotation file
        
        annotation_file = image_file[:-3]
        annotation_path = os.path.join(annotations_dir, annotation_file + 'txt')
        
        # Read the annotation file and extract the box coordinates
        with open(annotation_path, "r") as f:
            annotation_data = f.readlines()
        
        for annotation in annotation_data:
            label, x_center, y_center, width, height = annotation.split()
            x_min = int((float(x_center) - (float(width) / 2)) * image.shape[1])
            y_min = int((float(y_center) - (float(height) / 2)) * image.shape[0])
            x_max = int((float(x_center) + (float(width) / 2)) * image.shape[1])
            y_max = int((float(y_center) + (float(height) / 2)) * image.shape[0])
            
            # Draw the bounding box on the image
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            
        # Display the image with the bounding boxes
        cv2.imshow("Image with Annotations", image)
        cv2.waitKey(1)
        
    # Prompt user for input and write label to output file
    video_name = os.path.basename(os.path.normpath(folder_dir))
    output_file_path = os.path.join(output_dir, f"{video_name}.txt")
    if not os.path.exists(output_file_path):
        with open(output_file_path, "w") as f:
            pass # create empty file
    while True:
        user_input = input("Did the object cross the line? (y/n)")
        if user_input == "y":
            label = "1"
            break
        elif user_input == "n":
            label = "0"
            break
        else:
            print("Invalid input. Please enter 'y' or 'n'.")
    with open(output_file_path, "w") as f:
        f.write(label)
        
    # Clean up
    cv2.destroyAllWindows()
