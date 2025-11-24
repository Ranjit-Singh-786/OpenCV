import cv2
import numpy as np
import os
from ultralytics import YOLO
import cv2
import numpy as np

def enhance_face(img: np.ndarray) -> np.ndarray:

    if img is None or img.size == 0:
        return img
    

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Use a small tile grid (e.g., 4x4) and low clip limit (e.g., 2.0) for less aggressive enhancement
    clahe = cv2.createCLAHE(clipLimit=0.8, tileGridSize=(1,1)) 
    # clipLimit : Isko badhayein (e.g., 2.0 se 4.0 tak). Jitna zyada hoga, dark areas utne bright honge, lekin noise bhi badh sakta hai.
    cl = clahe.apply(l)
    
    # Merge the enhanced L channel back
    limg = cv2.merge((cl, a, b))
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    

    denoised_img = cv2.GaussianBlur(enhanced_img, (3,3), 0)
    
    # --- 3. Sharpening (Emphasize Edges) ---
    # Sharpening kernel to boost edges (eyes, nose, mouth) for feature extraction
    sharpening_kernel = np.array([[-1,-1,-1], 
                                  [-1, 9,-1],
                                  [-1,-1,-1]])
    sharpened_img = cv2.filter2D(denoised_img, -1, sharpening_kernel)

    # --- 4. Blending for Subtle Effect ---
    # Blend the sharpened image with the denoised image (e.g., 70% sharpened, 30% denoised)
    # This reduces the harshness of the sharpening effect, which can sometimes hurt recognition
    final_img = cv2.addWeighted(sharpened_img, 0.25, denoised_img, 0.75, 1)
    # last parameter: Isko 0 se badha kar 10, 20, ya 30 karein. Ye poori image mein light add kar dega.
    
    return final_img

video_path1 = r"C:\Users\s86\Desktop\NVR_ch1_main_20251120132931_20251120133407.mp4"

RAW_DIR = "raw_faces"
ENH_DIR = "enhanced_faces"

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(ENH_DIR, exist_ok=True)


model = YOLO(video_path)

cap = cv2.VideoCapture(r"\\SAGNAS\java\Ranjit Singh\AI_Project Video 01052025\NVR_ch1_main_20251016093942_20251016094002.mp4")

if not cap.isOpened():
    print("❌ Error: Cannot connect to video/RTSP stream")
    exit()

print("▶ Connected. Press 'q' to quit.")

frame_count = 0
save_count = 0
i = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    i += 1
    if i % 5 == 0:
        frame_count += 1

        results = model.track(
            frame,
            conf=0.5,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False
        )

        result = results[0]

        if result.boxes is not None and result.boxes.id is not None:

            boxes = result.boxes.xyxy.cpu().numpy()
            track_ids = result.boxes.id.cpu().numpy().astype(int)
            confidences = result.boxes.conf.cpu().numpy()

            for box, track_id, conf in zip(boxes, track_ids, confidences):

                x1, y1, x2, y2 = map(int, box)

                y1 = max(0, y1)
                x1 = max(0, x1)
                y2 = min(frame.shape[0], y2)
                x2 = min(frame.shape[1], x2)
                
                face_crop = frame[y1:y2, x1:x2]

                if face_crop.size == 0:
                    continue

                if face_crop.shape[0] < 60 or face_crop.shape[1] < 60:
                    continue

                save_count += 1

                raw_path = os.path.join(RAW_DIR, f"raw_{save_count}.jpg")
                cv2.imwrite(raw_path, face_crop)

                enhanced_face = enhance_face(face_crop)

                enh_path = os.path.join(ENH_DIR, f"enh_{save_count}.jpg")
                cv2.imwrite(enh_path, enhanced_face)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

