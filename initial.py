import cv2
import torch
import numpy as np
import tkinter as tk
from tkinter import filedialog

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

def categorize_vehicle(vehicle_box, frame_width):
    x_center = (vehicle_box[0] + vehicle_box[2]) / 2

    if x_center < frame_width / 3:
        return 'adjacent'
    elif x_center > 2 * frame_width / 3:
        return 'oncoming'
    else:
        return 'leading'

def select_video_file():
    root = tk.Tk()
    root.withdraw()
    video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov")])
    return video_path

def main():
    video_path = select_video_file()
    if not video_path:
        print("No video file selected.")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file.")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    output_path = 'output_video.avi'

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Detect vehicles
        results = model(frame)
        detections = results.xyxy[0].cpu().numpy()

        for detection in detections:
            x1, y1, x2, y2, conf, cls = detection
            if int(cls) == 2:  # Assuming class 2 is 'car'
                category = categorize_vehicle([x1, y1, x2, y2], frame_width)

                if category == 'leading':
                    color = (0, 0, 255)  # Red
                elif category == 'adjacent':
                    color = (0, 255, 255)  # Yellow
                elif category == 'oncoming':
                    color = (0, 255, 0)  # Green

                # Draw bounding box
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Processed video saved as {output_path}")

if __name__ == "__main__":
    main()
