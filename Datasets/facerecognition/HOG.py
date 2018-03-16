# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 18:14:37 2018

@author: Ravikiran.Tamiri
"""

import dlib
#import os
import imageio
import glob

#image_filenames = filter(lambda x: x.endswith('.jpg'), os.listdir('images/'))
image_filenames = glob.glob('B_images' + '/**/*.jpg', recursive=True)
image_filenames = sorted(image_filenames)
file_name = image_filenames[0]
paths_to_images = [x for x in image_filenames]

# Create a HOG face detector using the built-in dlib class
face_detector = dlib.get_frontal_face_detector()
win = dlib.image_window()
# Load the image into an array
image = imageio.imread(paths_to_images[0])
# Run the HOG face detector on the image data.
# The result will be the bounding boxes of the faces in our image.
detected_faces = face_detector(image, 1)

print("I found {} faces in the file {}".format(len(detected_faces), file_name))

# Open a window on the desktop showing the image
win.set_image(image)

# Loop through each face we found in the image
for i, face_rect in enumerate(detected_faces):

	# Detected faces are returned as an object with the coordinates 
	# of the top, left, right and bottom edges
	print("- Face #{} found at Left: {} Top: {} Right: {} Bottom: {}".format(i, face_rect.left(), face_rect.top(), face_rect.right(), face_rect.bottom()))

	# Draw a box around each face we found
	win.add_overlay(face_rect)
