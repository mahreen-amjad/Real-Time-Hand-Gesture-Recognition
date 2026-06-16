import cv2
from ultralytics import YOLO
import pyautogui
import time

# Official pre-trained model load 
model = YOLO('yolov8n.pt')

# Webcam Setup
cap = cv2.VideoCapture(0)
last_action_time = time.time()

# Cooldowns
COOLDOWN_SPACE = 2.0  # Play/Pause
COOLDOWN_VOL = 0.3    # Volume change 

print("====================================================")
# 
print("YOLOv8 Adaptive Spatial Gesture Control Activated...")
print("Press 'q' inside the webcam window to exit.")
print("====================================================")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # (mirror view)
    frame = cv2.flip(frame, 1)
    height, width, _ = frame.shape
    
    # YOLO prediction (stream=True and verbose=False to get speed max )
    results = model(frame, stream=True, verbose=False)
    
    gesture = "No Gesture Detected"
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0])
            
            # If model detect person (Class ID 0)
            if cls == 0:
                # Bounding box coordinates extract 
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Spatial Geometry (DIP)
                box_width = x2 - x1
                box_height = y2 - y1
                aspect_ratio = box_width / box_height
                
                current_time = time.time()
                
                # ----------------------------------------------------
                # GESTURE LOGIC BASED ON GEOMETRIC BOUNDING BOX
                # ----------------------------------------------------
                
                # 1. Open Palm Proxy (Wide Area Gesture)
                if aspect_ratio > 0.65:
                    gesture = "Detected: Open Palm -> Action: Play/Pause"
                    # Draw Green Box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    if current_time - last_action_time > COOLDOWN_SPACE:
                        pyautogui.press('space')
                        last_action_time = current_time
                        
                # 2. Volume Controllers (Vertical Spatial Tracking)
                else:
                    # Box in  upper half screen -> Thumbs Up Trigger
                    if y1 < height // 2:
                        gesture = "Detected: Thumbs Up -> Action: Volume UP"
                        # Draw Yellow Box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
                        if current_time - last_action_time > COOLDOWN_VOL:
                            pyautogui.press('volumeup')
                            last_action_time = current_time
                    
                    # If box is in lower half screen -> Thumbs Down Trigger
                    else:
                        gesture = "Detected: Thumbs Down -> Action: Volume DOWN"
                        # Draw Red Box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                        if current_time - last_action_time > COOLDOWN_VOL:
                            pyautogui.press('volumedown')
                            last_action_time = current_time

    # Display on feedback screen
    cv2.putText(frame, gesture, (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Grid lines draw so user understand separation (Optional DIP Feature)
    cv2.line(frame, (0, height // 2), (width, height // 2), (200, 200, 200), 1, cv2.LINE_AA)
    
    cv2.imshow("DIP Final Project - Gesture Controller", frame)

    # 'q' loop break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Project stopped successfully.")