'''
This creates frames from videos in 15fps such that it fits the anotations downloaded from cvat
'''


import cv2
import os

# Specify the input video file path
input_path = 'Data\\Trae\\fixed ones'

test= os.listdir(input_path)

# Get the input video file name (without extension)

for file in test:
    if file.endswith('.avi'):    
        input_name = file

        # Specify the output frames directory path
        output_path = os.path.join('Data\\Trae\\fixed ones\\blah', input_name[:-4])
        output_path_full = output_path + '\\obj_train_data'

        frame_rate = 15

        # Create the output directory if it doesn't exist
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        if not os.path.exists(output_path_full):
            os.makedirs(output_path_full)

        
        # Open the video file
        video_path = input_path + '\\' + input_name
        cap = cv2.VideoCapture(video_path)

        # Get the total number of frames in the video
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Calculate the time duration of the video (in seconds)
        duration = total_frames / cap.get(cv2.CAP_PROP_FPS)

        # Calculate the total number of frames to extract
        num_frames = int(duration * frame_rate)

        # Calculate the interval between frames (in seconds)
        interval = 1 / frame_rate

        # Loop through each frame and extract it as an image
        for i in range(num_frames):
            # Calculate the time offset for the current frame
            t = i * interval

            # Set the current time position in the video
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)

            # Read the current frame
            ret, frame = cap.read()

            # Save the frame as an image file
            frame_path = os.path.join(output_path_full, f'frame_{i:06d}.jpg')
            cv2.imwrite(frame_path, frame)

        # Release the video file
        cap.release()
