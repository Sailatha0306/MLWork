import os
import dlib
import imageio

predictor_model = "shape_predictor_68_face_landmarks.dat"

image_filenames = filter(lambda x: x.endswith('.jpg'), os.listdir('AtmecsImgs/'))
image_filenames = sorted(image_filenames)
file_name = image_filenames[0]
paths_to_images = ['AtmecsImgs/' + x for x in image_filenames]

# Create a HOG face detector using the built-in dlib class
face_detector = dlib.get_frontal_face_detector()
face_pose_predictor = dlib.shape_predictor(predictor_model)

win = dlib.image_window()

image = imageio.imread(paths_to_images[0])

detected_faces = face_detector(image, 1)

print("Found {} faces in the image file {}".format(len(detected_faces), file_name))

win.set_image(image)

for i, face_rect in enumerate(detected_faces):
	print("- Face #{} found at Left: {} Top: {} Right: {} Bottom: {}".format(i, face_rect.left(), face_rect.top(), face_rect.right(), face_rect.bottom()))
	win.add_overlay(face_rect)
	#pose_landmarks = face_pose_predictor(image, face_rect)
	#win.add_overlay(pose_landmarks)
