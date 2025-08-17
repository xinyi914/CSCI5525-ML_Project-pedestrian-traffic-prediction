'''
This unzips folders and creates new folders with the same name as the zip
'''

import os
import zipfile

# Specify the directory where the zip files are located
zip_dir = 'Data\\Hangyu\\'

# Get a list of all zip files in the directory
zip_files = [f for f in os.listdir(zip_dir) if f.endswith('.zip')]

# Loop through each zip file and extract its contents
for zip_file in zip_files:
    with zipfile.ZipFile(os.path.join(zip_dir, zip_file), 'r') as zip_ref:
        zip_ref.extractall(os.path.join(zip_dir, os.path.splitext(zip_file)[0]))