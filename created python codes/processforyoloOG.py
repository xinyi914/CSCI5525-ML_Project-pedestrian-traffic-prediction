import os
import shutil

"""
This is another data editing script that resturces the data to fit yolo but this was the orgianal atempt which did work but we wanted to build on it.
"""

# Path to the folder containing the image folders
src_folder = "Data"

# Path to the destination folder where you want to move the images and annotations
dst_folder = "OG_training_attempt_data"

frame_num = 1
# Loop through all the folders in the source folder
for folder_name in os.listdir(src_folder):
    name_dir = os.path.join(src_folder, folder_name)
    for name_folder in os.listdir(name_dir):
        folder_path = os.path.join(src_folder, folder_name)
        # Loop through all the files in the folder
        if 'site' in name_folder:
                # Check if the file is an image or annotation file
                obj_dir = os.path.join(folder_path, name_folder, 'obj_train_data')
                for fileidx in os.listdir(obj_dir):
                    if fileidx.endswith(".jpg") or fileidx.endswith(".png") or fileidx.endswith(".PNG"):
                        # Construct the new file name with a fixed prefix and a zero-padded frame number
                        new_file_name = "frame_" + str(frame_num).zfill(5) + ".jpg"
                        
                        # Increment the frame number counter
                        frame_num += 1
                        
                        # Copy the image file to the destination folder and rename it
                        src_file_path = os.path.join(obj_dir, fileidx)
                        dst_file_path = os.path.join(dst_folder, new_file_name)
                        shutil.copy(src_file_path, dst_file_path)
                    
                        
                    elif fileidx.endswith(".txt"):
                        # Construct the new file name for the annotation file
                        new_file_name = "frame_" + str(frame_num-1).zfill(5) + ".txt"
                        
                        # Copy the annotation file to the destination folder and rename it
                        src_file_path = os.path.join(obj_dir, fileidx)
                        dst_file_path = os.path.join(dst_folder, new_file_name)
                        shutil.copy(src_file_path, dst_file_path)