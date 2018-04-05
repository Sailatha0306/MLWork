import cv2

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,960)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,540)


while(True):
    # Capture frame-by-frame
    ret, framex = cap.read()
    
    frame = cv2.resize(framex,(960,540))
    
    cv2.waitKey(30)
    
    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('frame',gray)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()