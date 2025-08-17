'''
This iterates over folders of each site and copies the data to rename them to create one dataset
'''


import os
import shutil

# Set the source directory path containing the images and text files
src_dir_path = 'E:\\data\\sites'
test = os.listdir(src_dir_path)


# Set the destination directory path for the renamed images
dst_dir_path = 'E:\\data\\All_training_data'

# Set the prefix for the renamed files
prefix = 'Frame'

# Set the starting index for the renamed files
start_index = 1
for site in test:
    blah =os.path.join(src_dir_path, site)
    obj_folders = os.path.join(blah, 'obj_train_data')
    # Loop through all files in the source directory
    for filename in os.listdir(obj_folders):
        # Check if the file is an image or text file
        if filename.endswith('.jpg') or filename.endswith('.PNG') or filename.endswith('.jpeg'):
            # Generate the new file name using the prefix and index
            new_filename = f'{prefix}_{start_index:06d}.{filename.split(".")[-1]}'
            start_index += 1
            
            # Copy the image file to the destination directory with the new name
            shutil.copy(os.path.join(obj_folders, filename), os.path.join(dst_dir_path, new_filename))
            
            # Get the corresponding text file name
            txt_filename = filename.split('.')[0] + '.txt'
            
            # Generate the new text file name using the prefix and index
            new_txt_filename = f'{prefix}_{start_index-1:06d}.txt'
            
            # Copy the text file to the destination directory with the new name
            shutil.copy(os.path.join(obj_folders, txt_filename), os.path.join(dst_dir_path, new_txt_filename))

print(f'total images in data: {start_index}')