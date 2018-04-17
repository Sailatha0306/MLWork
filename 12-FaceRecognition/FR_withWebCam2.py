"""
Created on Tue Mar 27 16:46:47 2018

@author: Ravikiran.Tamiri
"""
import cv2
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
        return [adjust_tuple_bounds(conv_rect2tuple(face), img.shape) for face in face_detector(img, number_of_times_to_upsample)]
       
def conv_rect2tuple(rect):
    return rect.top(), rect.right(), rect.bottom(), rect.left()

def adjust_tuple_bounds(css, image_shape):
    return max(css[0], 0), min(css[1], image_shape[1]), min(css[2], image_shape[0]), max(css[3], 0)

def face_encodings(face_image, known_face_locations=None, num_jitters=1):
    raw_landmarks = _raw_face_landmarks(face_image, known_face_locations, model="small")
    return [np.array(face_encoder.compute_face_descriptor(face_image, raw_landmark_set, num_jitters)) for raw_landmark_set in raw_landmarks]

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


import time

input_video = cv2.VideoCapture(0)
input_video.set(3,1280)
input_video.set(4,1024)
input_video.set(15, 0.1)
time.sleep(2)

input_video.set(15, -8.0)

ret,frame = input_video.read()

name_img = "inputimgs/frame.jpg"
cv2.imwrite(name_img, frame)
#length = int(input_video.get(cv2.CAP_PROP_FRAME_COUNT))
#my_fps = input_video.get(cv2.CAP_PROP_FPS)
#my_fs_w = input_video.get(cv2.CAP_PROP_FRAME_WIDTH)
#my_fs_h = input_video.get(cv2.CAP_PROP_FRAME_HEIGHT)

image_filenames = glob.glob('Video_data\\myTestVideo\\images' + '\\*.jpg', recursive=True)
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

fcount = 0
fnum = 0 
while True:
    #read frame by frame
    ret,frame = input_video.read()
    fnum += 1
    
    name_img = "inputimgs/frame%d.jpg"%fnum
    cv2.imwrite(name_img, frame)
    
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    
     # Convert from BGR color ( OpenCV) to RGB color (face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    if fcount == 0:
        face_locs = face_locations(rgb_small_frame)
        inp_face_encodings = face_encodings(rgb_small_frame, face_locs)
        
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
        
    fcount +=1
    fcount = fcount%2
    
    for (top, right, bottom, left), name in zip(face_locs, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        
        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)
        
    
    #name_img = "frame%d.jpg"%fnum
    #cv2.imwrite(name_img, frame)
    # Display the resulting image
    cv2.imshow('Video', frame)
    
    
    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
input_video.release()
cv2.destroyAllWindows()