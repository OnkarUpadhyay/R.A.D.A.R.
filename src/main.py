import cv2
import csv
import sys
from ultralytics import YOLO
from anpr_utils import (
    preprocess_license_plate, 
    license_complies_format, 
    get_car, 
    read_license_plate
)

# ==========================================
# 1. Initialize Models
# ==========================================
print("Loading Models & EasyOCR Engine on GPU...")
coco_model = YOLO('yolov8n.pt') 
vehicles = [2, 3, 5, 7]

plate_model = YOLO('License_Plate_10k/weights/best.pt')

# ==========================================
# 2. Main Video Processing Loop
# ==========================================
video_path = sys.argv[1] if len(sys.argv) > 1 else 'sample_30fps.mp4'
cap = cv2.VideoCapture(video_path)

results = {} 
frame_nmr = -1
ret = True

print(f"Processing Video Frames from: {video_path}")
while ret:
    frame_nmr += 1
    ret, frame = cap.read()
    if not ret:
        break

    results[frame_nmr] = {}
    track_results = coco_model.track(frame, persist=True, classes=vehicles, verbose=False)
    plate_results = plate_model(frame, conf=0.35, verbose=False)
    
    for plate_box in plate_results[0].boxes:
        px1, py1, px2, py2 = plate_box.xyxy[0].cpu().numpy()
        bbox_score = float(plate_box.conf[0].cpu().numpy())
        
        car_coords, car_id = get_car((px1, py1, px2, py2), track_results)
        
        if car_id != -1:
            plate_crop = frame[int(py1):int(py2), int(px1):int(px2)]
            
            if plate_crop.size > 0:
                processed_crop = preprocess_license_plate(plate_crop)
                
                # Read text using EasyOCR on GPU (raw text)
                text, text_score = read_license_plate(processed_crop)
                
                if text and license_complies_format(text):
                    if text_score > 0.25:
                        results[frame_nmr][car_id] = {
                            'car': {'bbox': car_coords},
                            'license_plate': {
                                'bbox': [px1, py1, px2, py2],
                                'text': text,
                                'bbox_score': bbox_score,
                                'text_score': text_score
                            }
                        }

cap.release()

# ==========================================
# 3. Export to CSV
# ==========================================
print("Writing raw data to CSV...")
csv_path = 'test_results.csv'
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['frame_nmr', 'car_id', 'car_bbox', 'license_plate_bbox', 'license_plate_bbox_score', 'license_number', 'license_number_score'])
    
    for frame_nmr in results.keys():
        for car_id in results[frame_nmr].keys():
            data = results[frame_nmr][car_id]
            writer.writerow([
                frame_nmr, 
                car_id, 
                f"[{data['car']['bbox'][0]} {data['car']['bbox'][1]} {data['car']['bbox'][2]} {data['car']['bbox'][3]}]",
                f"[{data['license_plate']['bbox'][0]} {data['license_plate']['bbox'][1]} {data['license_plate']['bbox'][2]} {data['license_plate']['bbox'][3]}]",
                data['license_plate']['bbox_score'],
                data['license_plate']['text'],
                data['license_plate']['text_score']
            ])

print("Pipeline Complete! Output saved to test_results.csv")