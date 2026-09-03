import cv2
import mediapipe as mp
import torch
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from ViTModel import ViT

EMOTION_LABEL=['怒り','嫌悪','恐怖','喜び','悲しみ','驚き','無感情']


device=torch.device('cuda' if torch.cuda.is_available()else 'cpu')
vit_model=ViT(num_classes=7).to(device)
vit_model.load_state_dict(torch.load('vit_emotion_best.pth', map_location=device))
vit_model.eval()

def preprocess(face_crop):
    # BGRか->RGB
    face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
    #224*224にリサイズ
    face_resized = cv2.resize(face_rgb, (224, 224))
    tensor = face_resized.astype(np.float32) / 255.0
    tensor = np.transpose(tensor, (2, 0, 1))
    
    #正規化
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    tensor = (tensor - mean) / std
    
    #バッチ次元を追加
    tensor = torch.from_numpy(tensor).unsqueeze(0).to(device)
    return tensor

#MediaPipe Face Landmarkerの設定
model_path = 'face_landmarker.task'
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=True,
    num_faces=1,
    min_face_detection_confidence=0.5
)
detector = vision.FaceLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    detection_result = detector.detect(mp_image)

    if detection_result.face_landmarks:
        landmarks = detection_result.face_landmarks[0]
        h, w, _ = frame.shape

        #顔枠の算出
        x_coords = [int(lm.x * w) for lm in landmarks]
        y_coords = [int(lm.y * h) for lm in landmarks]
        x_min, x_max = max(0, min(x_coords)), min(w, max(x_coords))
        y_min, y_max = max(0, min(y_coords)), min(h, max(y_coords))

        #顔枠の描画
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

        #顔領域の切り出しとViT推論
        if (x_max - x_min) > 0 and (y_max - y_min) > 0:
            face_crop = frame[y_min:y_max, x_min:x_max]
            
            #前処理
            input_tensor = preprocess(face_crop)
            
            #ViT推論
            with torch.no_grad():
                output = vit_model(input_tensor)
                probabilities = torch.softmax(output, dim=1)
                pred_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][pred_class].item()

            #推論結果の上部表示
            label_text = f"ViT: {EMOTION_LABEL[pred_class]} ({confidence*100:.1f}%)"
            cv2.putText(frame, label_text, (x_min, max(20, y_min - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Emotion ViT Pipeline", frame)
    if cv2.waitKey(5) == 27:  # ESCキーで終了
        break

detector.close()
cap.release()
cv2.destroyAllWindows()
