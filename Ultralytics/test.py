import cv2
import numpy as np
from ultralytics import YOLO

video_path = "/home/21013187/Vietnam_Signlanguage_FE/vietnamsignlanguage/mediapipe/A52P1/rgb/197_A52P1_.avi"
output_path = "/home/21013187/Vietnam_Signlanguage_FE/masked_output.avi"
model_path = "/home/21013187/Vietnam_Signlanguage_FE/Ultralytics/yolo11n-seg.pt"

model = YOLO(model_path)

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
w, h = 640, 640

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

mask_writer = cv2.VideoWriter("/home/21013187/Vietnam_Signlanguage_FE/mask_output.avi",
                              fourcc, fps, (w, h), isColor=False)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (w, h))  # Resize mỗi frame

    results = model(frame)

    masks = results[0].masks
    if masks is not None and masks.data.shape[0] > 0:
        mask = masks.data[0].cpu().numpy()
        mask = (mask * 255).astype(np.uint8)
        mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))  # (W, H)
        mask_writer.write(mask)
        mask_3ch = cv2.merge([mask, mask, mask])
        if mask_3ch.dtype != frame.dtype:
            mask_3ch = mask_3ch.astype(frame.dtype)
        if frame.shape != mask_3ch.shape:
            print("❌ Shape không khớp:", frame.shape, mask_3ch.shape)
        else:
            person_only = cv2.bitwise_and(frame, mask_3ch)
    else:
        person_only = np.zeros_like(frame)

    out.write(person_only)

cap.release()
out.release()
print("✅ Đã lưu video tại:", output_path)
