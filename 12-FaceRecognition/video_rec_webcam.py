import cv2
cap = cv2.VideoCapture(0)
if cap.isOpened():
    print("Opened")
else:
    print("Not Opened")
cap.get(3)
cap.get(4)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Display the resulting frame
    cv2.imshow('frame',gray)
    #cv2.waitKey(20)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
