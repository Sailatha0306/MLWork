# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 14:41:03 2018

@author: Ravikiran.Tamiri
"""

import cv2
#import os
import glob
import numpy as np
import dlib
import imageio

TOLERANCE = 0.5
face_detector = dlib.get_frontal_face_detector()
pose_predictor_5_point = dlib.shape_predictor('shape_predictor_5_face_landmarks.dat')
face_encoder = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

shape_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

#face encodings (numbers to identify the face of a particular person)
face_recognition_model = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

def face_locations(img, number_of_times_to_upsample=1):
        return [_trim_css_to_bounds(_rect_to_css(face), img.shape) for face in face_detector(img, number_of_times_to_upsample)]
       
def _rect_to_css(rect):
    return rect.top(), rect.right(), rect.bottom(), rect.left()

def _trim_css_to_bounds(css, image_shape):
    return max(css[0], 0), min(css[1], image_shape[1]), min(css[2], image_shape[0]), max(css[3], 0)

def face_encodings(face_image, known_face_locations=None, num_jitters=1):
    raw_landmarks = _raw_face_landmarks(face_image, known_face_locations, model="small")
    return [np.array(face_encoder.compute_face_descriptor(face_image, raw_landmark_set, num_jitters)) for raw_landmark_set in raw_landmarks]

def _raw_face_landmarks(face_image, face_locations=None, model="large"):
    if face_locations is None:
        face_locations = face_detector(face_image,1)
    else:
        face_locations = [_css_to_rect(face_location) for face_location in face_locations]

    pose_predictor = shape_predictor
    return [pose_predictor(face_image, face_location) for face_location in face_locations]

def _css_to_rect(css):
    return dlib.rectangle(css[3], css[0], css[1], css[2])

def compare_face_encodings(known_faces, face):
    return (np.linalg.norm(known_faces-face, axis=1) <= TOLERANCE )

def get_face_encodings(path_to_image):
    image = imageio.imread(path_to_image)
    image = image[:,:,:3].copy()
    detected_faces = face_detector(image, 1)
    shapes_faces = [shape_predictor(image, face) for face in detected_faces]
    return [np.array(face_recognition_model.compute_face_descriptor(image, face_pose, 1)) for face_pose in shapes_faces]



input_movie = cv2.VideoCapture("Video_data/myTestVideo/input_clip.mp4")
#input_movie = cv2.VideoCapture("Video_data/set1/input_clip.mp4")

length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))
my_fps = input_movie.get(cv2.CAP_PROP_FPS)
my_fs_w = input_movie.get(cv2.CAP_PROP_FRAME_WIDTH)
my_fs_h = input_movie.get(cv2.CAP_PROP_FRAME_HEIGHT)


fourcc = cv2.VideoWriter_fourcc(*'XVID')

#output_movie = cv2.VideoWriter('output.avi', fourcc, 29.97, (640, 360))
#output_movie = cv2.VideoWriter('Video_data/set1/myTestoutput.mp4', fourcc, my_fps, (1920,1080))
output_movie = cv2.VideoWriter('Video_data/myTestVideo/output.avi', fourcc, 29.97, (1920,1080))


#lmm_image = face_recognition.load_image_file("lin-manuel-miranda.png")
#im = PIL.Image.open("lin-manuel-miranda.png")
#lm_face_encoding = get_face_encodings("lin-manuel-miranda.png")[0]
#al_face_encoding = get_face_encodings("alex-lacamoire.png")[0]
#b_obama_encoding = get_face_encodings("obama.jpg")[0]
#m_obama_encoding = get_face_encodings("mitchell_obama.jpg")[0]

image_filenames = glob.glob('Video_data\\myTestVideo\\images' + '\\*.jpg', recursive=True)
#image_filenames = glob.glob('Video_data\\set1\\images' + '\\*.jpg', recursive=True)
image_filenames = sorted(image_filenames)
paths_to_images = [ x for x in image_filenames]
image2name = {}
img_count = 0
known_faces_encodings =[]

for path_to_image in paths_to_images:
    face_encodings_in_image = get_face_encodings(path_to_image)
    if len(face_encodings_in_image) == 1:
        known_faces_encodings.append(get_face_encodings(path_to_image)[0])
        matched_name = path_to_image.split('\\')[3][:-4]
        image2name[img_count] = matched_name      
        img_count += 1
        

frame_no = 0

while True:
    ret,frame = input_movie.read()
    frame_no += 1
    
    if not ret:
        break
    
    #rgb_frame = frame[:,:,::-1]
    
    '''name = "frame%d.jpg"%frame_no
    cv2.imwrite(name, frame)'''

    face_locs = face_locations(frame)
    inp_face_encodings = face_encodings(frame, face_locs)
    
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
        
    for (top, right, bottom, left), name in zip(face_locs, face_names):
        if not name:
            continue
        
        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)
        
        
    
    name_img = "frame%d.jpg"%frame_no
    cv2.imwrite(name_img, frame)
    # Write the resulting image to the output video file
    print("Writing frame {} / {}".format(frame_no, length))
    output_movie.write(frame)
    

input_movie.release()
output_movie.release()
cv2.destroyAllWindows()
            