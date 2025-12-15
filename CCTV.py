import torch
import cv2
from PIL import Image
import clip
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# Load models only once
device = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5s.pt', force_reload=True)
clip_model, preprocess = clip.load("ViT-B/32", device=device)

def search_video(video_path, query):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    text_inputs = clip.tokenize([query]).to(device)
    found = False
    frame_skip = 5

    while cap.isOpened() and not found:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        if frame_id % frame_skip != 0:
            continue

        timestamp_sec = frame_id / fps

        results = model(frame[..., ::-1])
        detections = results.pandas().xyxy[0]

        for index, row in detections.iterrows():
            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            cropped_img = frame[y1:y2, x1:x2]

            if cropped_img.size == 0:
                continue

            image = Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
            image_input = preprocess(image).unsqueeze(0).to(device)

            with torch.no_grad():
                image_features = clip_model.encode_image(image_input)
                text_features = clip_model.encode_text(text_inputs)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                similarities = (image_features @ text_features.T)

            max_sim, max_idx = similarities.max(dim=1)
            if max_sim.item() > 0.3:
                found = True
                cap.release()
                return f"Found '{query}' at {timestamp_sec:.2f} seconds (Frame {frame_id})"

    cap.release()
    return "No matching object found in the video."

