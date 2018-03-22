# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 15:59:54 2018

@author: Ravikiran.Tamiri
This converts an image file into a set of face encodings corressponding to each face in the image.
"""
import PIL
import numpy as np
import dlib

image_file = "cric_team.jpg"
face_detector = dlib.get_frontal_face_detector()
#detects landmark points in faces and also the pose/angle of the face
shape_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
face_encoder = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

def img2np_array(image_file):
    im = PIL.Image.open(image_file)
    im = im.convert('RGB')
    return np.array(im)

def get_face_locations(img, number_of_times_to_upsample=1):
    #number_of_times_to_upsample: How many times to upsample the image looking for faces. Higher numbers find smaller faces.
    return [adjust_tuple_bounds(conv_rect2tuple(face), img.shape) for face in face_loc_arrays(img, number_of_times_to_upsample)]

def adjust_tuple_bounds(css, image_shape):
    return max(css[0], 0), min(css[1], image_shape[1]), min(css[2], image_shape[0]), max(css[3], 0)

def conv_rect2tuple(rect):
    return rect.top(), rect.right(), rect.bottom(), rect.left()

def conv_tuple2rect(css):
    return dlib.rectangle(css[3], css[0], css[1], css[2])

def face_loc_arrays(img, number_of_times_to_upsample=1):
        return face_detector(img, number_of_times_to_upsample)

def get_face_encodings(face_image, known_face_locations=None, num_jitters=1):
    face_locations = [conv_tuple2rect(face_location) for face_location in known_face_locations]
    raw_landmarks = [shape_predictor(face_image, face_location) for face_location in face_locations]
    return [np.array(face_encoder.compute_face_descriptor(face_image, raw_landmark_set, num_jitters)) for raw_landmark_set in raw_landmarks]


#1 - Convert an image file into a numpy array
img_array = img2np_array(image_file)
#2 - Using the numpy array, get the bounding boxes for all the faces within the image.
face_img_w_boxes = get_face_locations(img_array)
#3 - Get the face encoding for all the faces in this image.
face_encodings = get_face_encodings(img_array,face_img_w_boxes)

