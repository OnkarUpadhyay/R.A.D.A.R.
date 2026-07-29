import cv2
import pandas as pd
import numpy as np
import ast
import sys
import csv
from anpr_utils import interpolate_bounding_boxes

def draw_border(img, top_left, bottom_right, color=(0, 255, 0), thickness=6, line_length_x=50, line_length_y=50):
    x1, y1 = top_left
    x2, y2 = bottom_right
    cv2.line(img, (x1, y1), (x1, y1 + line_length_y), color, thickness)  
    cv2.line(img, (x1, y1), (x1 + line_length_x, y1), color, thickness)
    cv2.line(img, (x1, y2), (x1, y2 - line_length_y), color, thickness)  
    cv2.line(img, (x1, y2), (x1 + line_length_x, y2), color, thickness)
    cv2.line(img, (x2, y1), (x2 - line_length_x, y1), color, thickness)  
    cv2.line(img, (x2, y1), (x2, y1 + line_length_y), color, thickness)
    cv2.line(img, (x2, y2), (x2, y2 - line_length_y), color, thickness)  
    cv2.line(img, (x2, y2), (x2 - line_length_x, y2), color, thickness)
    return img

def render_anpr_video(input_video_path, csv_path, output_video_path):
    print(f"Rendering output video: {output_video_path}...")
    results = pd.read_csv(csv_path)

    cap = cv2.VideoCapture(input_video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # ==========================================
    # Get highest confidence plate image and text[cite: 20]
    # ==========================================
    license_plate = {}
    for car_id in np.unique(results['car_id']):
        max_ = np.amax(results[results['car_id'] == car_id]['license_number_score'])
        
        # Ensure we only process cars that actually had a successful OCR read (score > 0)
        if max_ > 0:
            license_plate[car_id] = {
                'license_crop': None,
                'license_plate_number': results[(results['car_id'] == car_id) & (results['license_number_score'] == max_)]['license_number'].iloc[0]
            }
            cap.set(cv2.CAP_PROP_POS_FRAMES, results[(results['car_id'] == car_id) & (results['license_number_score'] == max_)]['frame_nmr'].iloc[0])
            ret, frame = cap.read()

            if ret:
                x1, y1, x2, y2 = ast.literal_eval(results[(results['car_id'] == car_id) & (results['license_number_score'] == max_)]['license_plate_bbox'].iloc[0].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))
                
                # Crop and resize[cite: 20]
                license_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
                if license_crop.size > 0:
                    license_crop = cv2.resize(license_crop, (int((x2 - x1) * 200 / (y2 - y1)), 200)) # Scaled down slightly to fit standard 1080p safely
                    license_plate[car_id]['license_crop'] = license_crop

    frame_nmr = -1
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # ==========================================
    # Main render loop[cite: 20]
    # ==========================================
    ret = True
    while ret:
        ret, frame = cap.read()
        frame_nmr += 1
        if ret:
            df_ = results[results['frame_nmr'] == frame_nmr]
            for row_indx in range(len(df_)):
                car_id = df_.iloc[row_indx]['car_id']
                
                # Only draw if we successfully extracted a plate for this car
                if car_id in license_plate and license_plate[car_id]['license_crop'] is not None:
                    car_x1, car_y1, car_x2, car_y2 = ast.literal_eval(df_.iloc[row_indx]['car_bbox'].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))
                    draw_border(frame, (int(car_x1), int(car_y1)), (int(car_x2), int(car_y2)), (0, 255, 0), 6, line_length_x=50, line_length_y=50)

                    x1, y1, x2, y2 = ast.literal_eval(df_.iloc[row_indx]['license_plate_bbox'].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

                    license_crop = license_plate[car_id]['license_crop']
                    H, W, _ = license_crop.shape

                    try:
                        # Draw the crop[cite: 20]
                        frame[int(car_y1) - H - 40:int(car_y1) - 40, int((car_x2 + car_x1 - W) / 2):int((car_x2 + car_x1 + W) / 2), :] = license_crop

                        # Draw the white background box[cite: 20]
                        frame[int(car_y1) - H - 120:int(car_y1) - H - 40, int((car_x2 + car_x1 - W) / 2):int((car_x2 + car_x1 + W) / 2), :] = (255, 255, 255)

                        (text_width, text_height), _ = cv2.getTextSize(license_plate[car_id]['license_plate_number'], cv2.FONT_HERSHEY_SIMPLEX, 1.5, 4)

                        # Put the text[cite: 20]
                        cv2.putText(frame,
                                    str(license_plate[car_id]['license_plate_number']),
                                    (int((car_x2 + car_x1 - text_width) / 2), int(car_y1 - H - 65 + (text_height / 2))),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1.5,
                                    (0, 0, 0),
                                    4)
                    except:
                        pass # Handles out-of-bounds rendering gracefully[cite: 20]

            out.write(frame)

    out.release()
    cap.release()
    print("Video rendering successfully finished!")

if __name__ == '__main__':
    input_vid = sys.argv[1] if len(sys.argv) > 1 else 'sample_traffic_video.mp4'
    
    print("Loading CSV and interpolating gaps...")
    with open('test_results.csv', 'r') as file:
        reader = csv.DictReader(file)
        data = list(reader)

    final_data = interpolate_bounding_boxes(data)
    
    print("Saving interpolated data...")
    header = ['frame_nmr', 'car_id', 'car_bbox', 'license_plate_bbox', 'license_plate_bbox_score', 'license_number', 'license_number_score']
    with open('test_results_interpolated.csv', 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=header)
        writer.writeheader()
        writer.writerows(final_data)
    
    render_anpr_video(input_vid, 'test_results_interpolated.csv', 'output_anpr_final.mp4')