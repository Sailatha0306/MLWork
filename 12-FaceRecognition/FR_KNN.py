"""
Created on Thu Mar 15 17:47:11 2018

@author: Ravikiran.Tamiri
"""

import os
import glob
import numpy as np
import dlib
import imageio
from sklearn import neighbors
import pickle

train_dir = 'data'
# to detect faces in images
face_detector = dlib.get_frontal_face_detector()

#detects landmark points in faces and also the pose/angle of the face
shape_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

#face encodings (numbers to identify the face of a particular person)
face_recognition_model = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

# To avoid false matches, use lower value
TOLERANCE = 0.5

def face_locations(img, number_of_times_to_upsample=1):
        return [_trim_css_to_bounds(_rect_to_css(face), img.shape) for face in face_detector(img, number_of_times_to_upsample)]

def _rect_to_css(rect):
    return rect.top(), rect.right(), rect.bottom(), rect.left()

def _trim_css_to_bounds(css, image_shape):
    return max(css[0], 0), min(css[1], image_shape[1]), min(css[2], image_shape[0]), max(css[3], 0)

def get_face_encodings(path_to_image):
    image = imageio.imread(path_to_image)
    image = image[:,:,:3].copy()
    detected_faces = face_detector(image, 1)
    shapes_faces = [shape_predictor(image, face) for face in detected_faces]
    return [np.array(face_recognition_model.compute_face_descriptor(image, face_pose, 1)) for face_pose in shapes_faces]

#train the classifier
X = []
y = []

#loop through the images
for class_dir in os.listdir(train_dir):
    if not os.path.isdir(os.path.join(train_dir, class_dir)):
            continue
    
    images_files = glob.glob('data\\' + class_dir + '/**/*.jpg', recursive=True)
    for img_file in images_files:
        face_encodings_in_image = get_face_encodings(img_file)
        
        if len(face_encodings_in_image) == 1:
            X.append(get_face_encodings(img_file)[0])
            y.append(class_dir)
            print("Processing file : "+img_file + " Class dir : " + class_dir)
            
# Create and train the KNN classifier
knn_clf = neighbors.KNeighborsClassifier(n_neighbors=5)
knn_clf.fit(X, y)

saved_path = r"D:\Others\Machine Learning A-Z Template Folder\Datasets\models\knn.p"
#saved_path = r"C:\Users\ravikiran.tamiri\Desktop"
#
# Save the trained KNN classifier
if saved_path is not None:
        with open(saved_path, 'wb') as f:
            pickle.dump(knn_clf, f)
 
#get the test files       
test_filenames = filter(lambda x: x.endswith('.jpg'), os.listdir('test/'))
paths_to_test_images = ['test/' + x for x in test_filenames]

for path_to_image in paths_to_test_images:
    face_encodings_in_image = get_face_encodings(path_to_image)
    image = imageio.imread(path_to_image)
    X_face_locations = face_locations(image)
    if len(face_encodings_in_image) == 1:
        closest_distances = knn_clf.kneighbors(face_encodings_in_image, n_neighbors=1)
        are_matches = [closest_distances[0][i][0] <= TOLERANCE for i in range(len(X_face_locations))]
        if are_matches[0]:
            pred = knn_clf.predict(face_encodings_in_image)
            #loc = X_face_locations
            print("Image : "+path_to_image+" is:>> " + pred[0])
        else:
            print("Image : "+path_to_image+" is NOT FOUND")
            