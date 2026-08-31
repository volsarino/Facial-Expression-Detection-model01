import mediapipe as mp
import cv2
#print(mp.__version__)
#print(cv2.__version__)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)

while cap.isOpened():
    success,frame=cap.read()
    if not success:
        break
    cv2.imshow("preview",frame)
    if cv2.waitKey(5)==27:
        break

cap.release()
cv2.destroyAllWindows()