import os
import shutil

"""
This code takes data from a folder and puts it in the format for better training in the RNN basicly 1 person per annotation
To further process this look at add_cross_label.py this one is watching all the videos and labeling if that class(person) crossed the street
"""


# Set input and output directories
input_dir = 'Data\\Trae'
output_dir = 'E:\\data\\Test_RNN_data'

# Loop through clip directories in input directory
for clip_dir in os.listdir(input_dir):
    if "site" in clip_dir:
        print(clip_dir)  
        clip_dir = os.path.join(input_dir, clip_dir)
        if os.path.isdir(clip_dir):
            # Create images/ and annotations/ subdirectories in output directory
            out_dir = os.path.join(output_dir, os.path.basename(clip_dir))


            # Loop through annotation files in input directory
            ann_dir = os.path.join(clip_dir, 'obj_train_data')
            for ann_file in os.listdir(ann_dir):
                if ann_file.endswith('.txt'):
                    # Modify class labels in annotation file
                    ann_path = os.path.join(ann_dir, ann_file)
                    with open(ann_path, 'r') as f:
                        ann_lines = f.readlines()
                    for i, line in enumerate(ann_lines):
                        line = line.strip().split()
                        if line[0] == '1':
                            # Delete line if class is 1
                            del ann_lines[i]
                        elif line[0] == '2':
                            # Change class label to 0 if class is 2
                            line[0] = '0'
                            ann_lines[i] = ' '.join(line) + '\n'
                    # Create new site directories for each line in annotation file
                    for i, line in enumerate(ann_lines):
                        line = line.strip().split()
                        person_dir = f'{out_dir}_person{i+1}'
                        
                        
                        os.makedirs(person_dir, exist_ok=True)
                        image_path = os.path.join(person_dir, 'images')
                        os.makedirs(image_path, exist_ok=True)
                        ano_path = os.path.join(person_dir, 'annotations')
                        os.makedirs(ano_path, exist_ok=True)



                        # Copy image file to images/ subdirectory in output directory
                        img_extensions = ['.jpg', '.jpeg', '.png']
                        for ext in img_extensions:
                            img_file = os.path.splitext(ann_file)[0] + ext
                            img_path = os.path.join(clip_dir, 'obj_train_data', img_file)
                            if os.path.isfile(img_path):
                                shutil.copy(img_path, os.path.join(person_dir, 'images', img_file))
                                break

                        
                        ann_file_new = os.path.splitext(ann_file)[0] + '.txt'
                        with open(os.path.join(person_dir, 'annotations', ann_file_new), 'w') as f:
                            f.writelines(ann_lines[i])
