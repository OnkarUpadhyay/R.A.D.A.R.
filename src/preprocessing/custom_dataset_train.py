from ultralytics import YOLO


def main():
    # 1. Load a pre-trained YOLOv8 nano model
    model = YOLO('yolov8n.pt')

    # 2. Train the model on your custom dataset
    # Note: Set device='cpu' if you do not have an NVIDIA GPU, or device=0 to use your GPU.
    results = model.train(
        data=r'C:\R.A.D.A.R\License Plate Recognition.v11i.yolov8\data.yaml',
        epochs=100,             # 50-100 epochs is a great starting point
        imgsz=640,             # Standard image size for YOLO
        batch=16,              # Adjust down to 8 if you get an Out of Memory (OOM) error
        name='License_Plate_10k',  # Names the folder where your results save
        device=0,
        workers=8,            # Safer on Windows; avoids multiprocessing spawn issues
    )
    return results


if __name__ == '__main__':
    main()