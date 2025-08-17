import os
import shutil
"""
This code was for the first go around at RNN training 
look at datasplit_individual and add_cross_label for the expanded version
"""
# Set input and output directories
input_dir = 'Data/Xinyi'
output_dir = 'RNN_data'

# Loop through clip directories in input directory
for clip_dir in os.listdir(input_dir):
    if "site" in clip_dir:
        print(clip_dir)  
        clip_dir = os.path.join(input_dir, clip_dir)
        if os.path.isdir(clip_dir):
            # Create images/ and annotations/ subdirectories in output directory
            out_dir = os.path.join(output_dir, os.path.basename(clip_dir))
            os.makedirs(os.path.join(out_dir, 'images'), exist_ok=True)
            os.makedirs(os.path.join(out_dir, 'annotations'), exist_ok=True)

            # Copy image and annotation files to appropriate directories in output directory
            obj_dir = os.path.join(clip_dir, 'obj_train_data')
            for filename in os.listdir(obj_dir):
                if filename.endswith('.jpg'):
                    # Copy image file to images/ subdirectory in output directory
                    shutil.copy(os.path.join(obj_dir, filename),
                                os.path.join(out_dir, 'images', filename))
                elif filename.endswith('.txt'):
                    # Copy annotation file to annotations/ subdirectory in output directory
                    shutil.copy(os.path.join(obj_dir, filename),
                                os.path.join(out_dir, 'annotations', filename))



