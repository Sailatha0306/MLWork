# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 14:41:03 2018

@author: Ravikiran.Tamiri
"""

import cv2
import glob
import numpy as np
import dlib
import imageio

print("\n####SETTING THE INITIAL PARAMETERES####\n")
TOLERANCE = 0.5
face_detector = dlib.get_frontal_face_detector()
#pose_predictor_5_point = dlib.shape_predictor('shape_predictor_5_face_landmarks.dat')
shape_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

#face encodings (numbers to identify the face of a particular person)
face_recognition_model = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

print("\n####FUNCTION DEFINITIONS####\n")
def face_locations(img, number_of_times_to_upsample=1):
        return [adjust_tuple_bounds(conv_rect2tuple(face), img.shape) for face in face_detector(img, number_of_times_to_upsample)]
       
def conv_rect2tuple(rect):
    return rect.top(), rect.right(), rect.bottom(), rect.left()

def adjust_tuple_bounds(css, image_shape):
    return max(css[0], 0), min(css[1], image_shape[1]), min(css[2], image_shape[0]), max(css[3], 0)

def face_encodings(face_image, known_face_locations=None, num_jitters=1):
    raw_landmarks = _raw_face_landmarks(face_image, known_face_locations, model="small")
    return [np.array(face_recognition_model.compute_face_descriptor(face_image, raw_landmark_set, num_jitters)) for raw_landmark_set in raw_landmarks]

def _raw_face_landmarks(face_image, face_locations=None, model="large"):
    if face_locations is None:
        face_locations = face_detector(face_image,1)
    else:
        face_locations = [conv_tuple2rect(face_location) for face_location in face_locations]

    pose_predictor = shape_predictor
    return [pose_predictor(face_image, face_location) for face_location in face_locations]

def conv_tuple2rect(css):
    return dlib.rectangle(css[3], css[0], css[1], css[2])

def compare_face_encodings(known_faces, face):
    return (np.linalg.norm(known_faces-face, axis=1) <= TOLERANCE )

def get_face_encodings(path_to_image):
    image = imageio.imread(path_to_image)
    image = image[:,:,:3].copy()
    detected_faces = face_detector(image, 1)
    shapes_faces = [shape_predictor(image, face) for face in detected_faces]
    return [np.array(face_recognition_model.compute_face_descriptor(image, face_pose, 1)) for face_pose in shapes_faces]


print("\n####READING THE IMAGES FROM TRAINING DATA####\n")
image_filenames = glob.glob('images' + '\\*.jpg', recursive=True)
image_filenames = sorted(image_filenames)
paths_to_images = [ x for x in image_filenames]
image2name = {}
img_count = 0
known_faces_encodings =[]

print("\n####GETTING THE FACE ENCODINGS####\n")
for path_to_image in paths_to_images:
    face_encodings_in_image = get_face_encodings(path_to_image)
    if len(face_encodings_in_image) == 1:
        known_faces_encodings.append(get_face_encodings(path_to_image)[0])
        matched_name = path_to_image.split('\\')[1][:-4]
        image2name[img_count] = matched_name      
        img_count += 1
        
print("\n####FACE ENCODING COMPLETED####\n")

print("\n####VIDEO CAPTURE FROM WEBCAM STARTED####\n")
#input_movie = cv2.VideoCapture("input_clip.mp4")
input_movie = cv2.VideoCapture(0)
frame_no = 0
process_this_frame = True

print("\n####READING THE IMAGES FROM WEBCAM####\n")
while True:
    ret,frame = input_movie.read()
    frame_no += 1
    
    if not ret:
        break
    
    small_frame = cv2.resize(frame,(0,0),fx=0.25,fy=0.25)
    
    rgb_frame = small_frame[:,:,::-1]
    
    if process_this_frame:
        face_locs = face_locations(rgb_frame)
        inp_face_encodings = face_encodings(rgb_frame, face_locs)
    
        face_names = []
        for face_encoding in inp_face_encodings:
            matches = compare_face_encodings(known_faces_encodings, inp_face_encodings[0])
            name = "Unknown"
            count = 0
        
            for match in matches:
                if match:
                    name = image2name[count]
                count += 1
            
            face_names.append(name)
            #if name == "Unknown":
                #ret = input("UNKNOWN FACE IDENTIFIED, WOULD LIKE TO ADD TO THE TRAINING DATA?(Y/N)")
                #if ret == "Y":
                    #name_img = input("ENTER THE NAME OF THE PERSON")
                    #name_img = "images\\" + name_img
                    #cv2.imwrite(name_img, rgb_frame)
        
    process_this_frame = not process_this_frame

    for (top, right, bottom, left), name in zip(face_locs, face_names):
        if not name:
            continue
        
        top *=4
        right*=4
        bottom*=4
        left*=4
        
        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    # Write the resulting image to the output video file
    cv2.imshow('Video', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

input_movie.release()
cv2.destroyAllWindows()