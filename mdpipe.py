import mediapipe as mp
import cv2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

model_path='face_landmarker.task'
#Face Landmarkerの設定
base_options=python.BaseOptions(model_asset_path=model_path)
options=vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=True,#ブレンドシェイプを出力
    num_faces=1,
    min_face_detection_confidence=0.5
)
# Face Landmarkerを作成
detector=vision.FaceLandmarker.create_from_options(options)

#webカメラの設定
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)

while cap.isOpened():
    success,frame=cap.read()
    if not success:
        break
    #BGR->RGB
    rgb_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    mp_image=mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb_frame)
    #ランドマーク検出
    detection_result=detector.detect(mp_image)

    #検出した顔を描画
    if detection_result.face_landmarks:
        landmarks=detection_result.face_landmarks[0]
        h,w,_=frame.shape
        #ランドマークの座標より顔の枠を算出
        x_coords=[int(lm.x*w)for lm in landmarks]
        y_coords=[int(lm.y*h)for lm in landmarks]
        #枠の端を取得   
        x_min,x_max=max(0,min(x_coords)),min(w,max(x_coords))
        y_min,y_max=max(0,min(y_coords)),min(h,max(y_coords))
        #枠を描画
        cv2.rectangle(frame,(x_min,y_min),(x_max,y_max),(255,0,0),2)

    cv2.imshow("preview",frame)
    if cv2.waitKey(5)==27:
        break
detector.close()
cap.release()
cv2.destroyAllWindows()