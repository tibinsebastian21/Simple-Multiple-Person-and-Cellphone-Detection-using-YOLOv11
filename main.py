import cv2
from ultralytics import YOLO

# Load the YOLOv11 model
model = YOLO("yolo11n.pt")  # Ensure you have the correct path to the model

# Set up video capture (0 for webcam, or replace with video file path)
cap = cv2.VideoCapture(0)

# Real-time inference loop
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    results = model(frame)  # Run inference on the current frame
    
    # Display results on the frame
    annotated_frame = results[0].plot()  # Annotate the frame with detection results
    
    # Initialize counters
    person_count = 0
    cellphone_detected = False

    # Check the results for persons and cellphones
    for result in results[0].boxes:
        if result.conf > 0.60:  # Check confidence score
            if result.cls == 0:  # Assuming class 0 is 'person'
                person_count += 1
            elif result.cls == 67:  # Assuming class 67 is 'cellphone' (check your model's class mapping)
                cellphone_detected = True

    # Display warning if conditions are met
    if person_count > 1 or cellphone_detected:
        cv2.putText(annotated_frame, "WARNING: Multiple persons or cellphone detected!", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Real-Time Object Detection", annotated_frame)  # Show the annotated frame
    
    # Check for 'q' key press to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()