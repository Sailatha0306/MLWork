This program is developed based on the below specifications.

Python - 3.6.4
dlib - 19.4.0
OpenCV - 3.2.0
Windows 7

How to run the program:
$$python video_FR.py

Result: This is turn on your webcam and able to identify the face in the frame either as Unknown or Name from the images directory(if present).
NOTE: Press q to end the program.


PreTrained Models:
We need the pre trained face recognition models which are stored in the form of dat files given below.
1. shape_predictor_68_face_landmarks.dat
2. dlib_face_recognition_resnet_model_v1.dat

These pre trained models are freely available on dlib site. 
Initially, we use get_frontal_face_detector() which helps to identify the human faces in the image we supplied. For simplicity, we trained with images which has only one face. 
Then, we used face landmarking model to align faces to a standard pose. 
Finally we use a model for face recognition.

Data building functions:
We iterate over each image from the training data to get the corresponding numpy array for that image.
Using our face detector from the 1st model, we mark the location of the face with in the image with top,right,bottom,left positions.
Using our 2nd model, we identify the shape/pose for that face.
Using our 3rd model, we get the face encodings (numpy array) for each pose of the face.
  
Reading the training data images:
Using the above data building functions, we build the face encoding for our problem specific training data.
We build an array of list of names corresponding to the image given.

Running the model:
We capture the frame from the webcam and send it for processing.
Using this frame(image), we identify the face location with in the frame, then the face pose and get the face encodings.
We compare the current image's encoding with that of face encodings available with us in the array.
If there is a match found, we mark the face in the frame with a red box and the corresponding name under it.
If there is no match found, we name it "Unknown".
We run this model in a never ending continuous while loop.