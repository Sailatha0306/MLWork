# -*- coding: utf-8 -*-
"""
Created on Sun Mar 11 22:56:39 2018

@author: Ravikiran.Tamiri
"""

import dlib
import numpy as np
import os
import imageio
import glob

# to detect faces in images
face_detector = dlib.get_frontal_face_detector()

#detects landmark points in faces and also the pose/angle of the face
shape_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

#face encodings (numbers to identify the face of a particular person)
face_recognition_model = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

# To avoid false matches, use lower value
# To avoid false negatives
TOLERANCE = 0.6

# ip-- image
# op--face encodings using the neural network
def get_face_encodings(path_to_image):
    #image = imageio.imread('B_images\AamairKhan\\3Idiots\images\Amir_1.jpg')
    image = imageio.imread(path_to_image)
    #image = imageio.imread('B_images\AamairKhan\\3Idiots\images\Amir_17.jpg')
    detected_faces = face_detector(image, 1)
    shapes_faces = [shape_predictor(image, face) for face in detected_faces]
    return [np.array(face_recognition_model.compute_face_descriptor(image, face_pose, 1)) for face_pose in shapes_faces]


def compare_face_encodings(known_faces, face):
    return (np.linalg.norm(known_faces - face, axis=1) <= TOLERANCE)

def find_match(known_faces, image2name, input_face):
    matches1 = compare_face_encodings(known_faces, input_face)
    #print(">>>>>>>>>>>>>>>"+matches)
    # Return the name of the first match
    count = 0
    for match1 in matches1:
        if match1:
            return image2name[count]
        count += 1
    # Return not found if no match found
    return 'Not Found'


image_filenames = glob.glob('B_images' + '/**/*.jpg', recursive=True)
image_filenames = sorted(image_filenames)
paths_to_images = [ x for x in image_filenames]
face_encodings = []

image2name = {}
#names = [x[:-6] for x in image_filenames]
#matched_name = names.split('\\')[4]
img_count = 0

for path_to_image in paths_to_images:
    face_encodings_in_image = get_face_encodings(path_to_image)
    if len(face_encodings_in_image) == 1:
        face_encodings.append(get_face_encodings(path_to_image)[0])
        matched_name = path_to_image.split('\\')[2]
        image2name[img_count] = matched_name
#        if matched_name in names2image:
#            names2image[matched_name].append(img_count)
#        else:
#            names2image.setdefault(img_count,[])
#            names2image[matched_name].append(img_count)
        
        img_count += 1
    else:
        print("Image must have only one face, check the image: " + path_to_image)



test_filenames = filter(lambda x: x.endswith('.jpg'), os.listdir('test/'))
paths_to_test_images = ['test/' + x for x in test_filenames]

for path_to_image in paths_to_test_images:
    face_encodings_in_image = get_face_encodings(path_to_image)
    if len(face_encodings_in_image) != 1:
        print("Image must have only one face, check the image: " + path_to_image)
    match = find_match(face_encodings, image2name, face_encodings_in_image[0])
    print(path_to_image, match)
    
    