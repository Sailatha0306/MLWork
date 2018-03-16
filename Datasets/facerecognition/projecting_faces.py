import dlib,os
import cv2
import openface

predictor_model = "shape_predictor_68_face_landmarks.dat"

image_filenames = filter(lambda x: x.endswith('.jpg'), os.listdir('images/'))
image_filenames = sorted(image_filenames)
file_name = image_filenames[0]
paths_to_images = ['images/' + x for x in image_filenames]

# Create a HOG face detector using the built-in dlib class
face_detector = dlib.get_frontal_face_detector()
face_pose_predictor = dlib.shape_predictor(predictor_model)
face_aligner = openface.AlignDlib(predictor_model)

image = cv2.imread(paths_to_images[0])

detected_faces = face_detector(image, 1)

print("Found {} faces in the image file {}".format(len(detected_faces), file_name))

for i, face_rect in enumerate(detected_faces):

#	print("- Face #{} found at Left: {} Top: {} Right: {} Bottom: {}".format(i, face_rect.left(), face_rect.top(), face_rect.right(), face_rect.bottom()))
	pose_landmarks = face_pose_predictor(image, face_rect)
	# Use openface to calculate and perform the face alignment
	alignedFace = face_aligner.align(534, image, face_rect, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
	# Save the aligned image to a file
	#cv2.imwrite("aligned_face_{}.jpg".format(i), alignedFace)