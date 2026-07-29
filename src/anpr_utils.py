import cv2
import csv
import string
import numpy as np
from scipy.interpolate import interp1d
import easyocr

# ==========================================
# 1. Initialize EasyOCR (GPU Accelerated)
# ==========================================
reader = easyocr.Reader(['en'], gpu=True)

# ==========================================
# 2. Image Preprocessing & OCR Reader
# ==========================================
def preprocess_license_plate(plate_crop):
    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(resized)
    blurred = cv2.GaussianBlur(enhanced, (3,3), 0)
    return blurred

def read_license_plate(license_plate_crop):
    """
    Executes EasyOCR inference on GPU and returns cleaned text and score.
    Confusion dictionaries and rigid formatting rules have been removed.
    """
    try:
        detections = reader.readtext(license_plate_crop)
        if not detections:
            return None, 0.0
        
        best_text = ""
        best_score = 0.0
        for bbox, text, score in detections:
            if score > best_score:
                best_score = score
                best_text = text
                
        # Clean text: keep only alphanumeric characters converted to uppercase
        clean_text = "".join(char for char in best_text if char.isalnum()).upper()
        if len(clean_text) >= 4 and len(clean_text) <= 10:
            return clean_text, float(best_score)
        
        return None, 0.0
    except Exception as e:
        return None, 0.0

def license_complies_format(text):
    clean_text = "".join(char for char in text if char.isalnum())
    if len(clean_text) < 4 or len(clean_text) > 10:
        return False
    return True

# ==========================================
# 3. Helper Function: Assign Plate to Car
# ==========================================
def get_car(plate, track_results):
    px1, py1, px2, py2 = plate
    if track_results[0].boxes.id is None:
        return -1, -1

    for i in range(len(track_results[0].boxes)):
        x1, y1, x2, y2 = track_results[0].boxes.xyxy[i].cpu().numpy()
        car_id = int(track_results[0].boxes.id[i].cpu().numpy())
        if px1 >= x1 and py1 >= y1 and px2 <= x2 and py2 <= y2:
            return (x1, y1, x2, y2), car_id
    return -1, -1 

# ==========================================
# 4. Interpolation Logic
# ==========================================
def interpolate_bounding_boxes(data):
    frame_numbers = np.array([int(row['frame_nmr']) for row in data])
    car_ids = np.array([int(float(row['car_id'])) for row in data])
    
    car_bboxes = np.array([list(map(float, row['car_bbox'].replace('[', '').replace(']', '').split())) for row in data])
    license_plate_bboxes = np.array([list(map(float, row['license_plate_bbox'].replace('[', '').replace(']', '').split())) for row in data])

    interpolated_data = []
    unique_car_ids = np.unique(car_ids)
    
    for car_id in unique_car_ids:
        frame_numbers_ = [p['frame_nmr'] for p in data if int(float(p['car_id'])) == int(float(car_id))]

        car_mask = car_ids == car_id
        car_frame_numbers = frame_numbers[car_mask]
        car_bboxes_interpolated = []
        license_plate_bboxes_interpolated = []

        first_frame_number = car_frame_numbers[0]

        for i in range(len(car_bboxes[car_mask])):
            frame_number = car_frame_numbers[i]
            car_bbox = car_bboxes[car_mask][i]
            license_plate_bbox = license_plate_bboxes[car_mask][i]

            if i > 0:
                prev_frame_number = car_frame_numbers[i-1]
                prev_car_bbox = car_bboxes_interpolated[-1]
                prev_license_plate_bbox = license_plate_bboxes_interpolated[-1]

                if frame_number - prev_frame_number > 1:
                    frames_gap = frame_number - prev_frame_number
                    x = np.array([prev_frame_number, frame_number])
                    x_new = np.linspace(prev_frame_number, frame_number, num=frames_gap, endpoint=False)
                    
                    interp_func = interp1d(x, np.vstack((prev_car_bbox, car_bbox)), axis=0, kind='linear')
                    interpolated_car_bboxes = interp_func(x_new)
                    
                    interp_func = interp1d(x, np.vstack((prev_license_plate_bbox, license_plate_bbox)), axis=0, kind='linear')
                    interpolated_license_plate_bboxes = interp_func(x_new)

                    car_bboxes_interpolated.extend(interpolated_car_bboxes[1:])
                    license_plate_bboxes_interpolated.extend(interpolated_license_plate_bboxes[1:])

            car_bboxes_interpolated.append(car_bbox)
            license_plate_bboxes_interpolated.append(license_plate_bbox)

        for i in range(len(car_bboxes_interpolated)):
            frame_number = first_frame_number + i
            row = {}
            row['frame_nmr'] = str(frame_number)
            row['car_id'] = str(car_id)
            row['car_bbox'] = f"[{' '.join(map(str, car_bboxes_interpolated[i]))}]"
            row['license_plate_bbox'] = f"[{' '.join(map(str, license_plate_bboxes_interpolated[i]))}]"

            if str(frame_number) not in frame_numbers_:
                row['license_plate_bbox_score'] = '0'
                row['license_number'] = '0'
                row['license_number_score'] = '0'
            else:
                original_row = [p for p in data if int(p['frame_nmr']) == frame_number and int(float(p['car_id'])) == int(float(car_id))][0]
                row['license_plate_bbox_score'] = original_row.get('license_plate_bbox_score', '0')
                row['license_number'] = original_row.get('license_number', '0')
                row['license_number_score'] = original_row.get('license_number_score', '0')

            interpolated_data.append(row)

    return interpolated_data