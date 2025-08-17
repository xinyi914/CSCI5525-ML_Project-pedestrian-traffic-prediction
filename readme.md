This project is for detecting whether the pedestiran will cross the road or not using RNN.
There weren't any extereme packages used outside of the ones we used in class like pytorch, numpy, cv2

Everything is found in the google drive at this link
https://drive.google.com/drive/folders/1RMgEGEFa2c3N97gecLLX6RwztELla4LE?usp=share_link

Everything is in a folder structure:
    assignments: everything we turned in
    NewVideoFram: videos after anotations
    OldVideoFrames: videos before anotations
    paper for reference: Papers we were deciding from
    Test_RNN_data: unzipped version of Test_RNN_data.zip #note both are the same just one is zipped that is the how the data was stored for RNN training purposes
    The meat of the project: This will have all of our code and stuffs
    
To run an example of the code working from start to finish go to The meat of the project -> Yolo -> yolov7-main_main -> and run fullsciript.py

This script will use test video found in the test_vids folder and extract anotations from it for the first x number of frames to create a sequence this goes through processing. It is then ran through the trained RNN and will output a if they will cross or not (1, 0)

 We decided to leave in code to to show the yolo bounding boxes on the prediction for demonstration purposes but that could be removed to make the code run smoother


