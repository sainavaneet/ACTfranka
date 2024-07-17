import cv2

# Initialize the camera (0 is usually the default camera)
cap = cv2.VideoCapture(0)

# Set the resolution to 480x640
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Check if CUDA is available and use GPU acceleration if possible
use_cuda = cv2.cuda.getCudaEnabledDeviceCount() > 0
print(f"CUDA is {'available' if use_cuda else 'not available'}.")

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # If the frame is read correctly, ret is True
    if not ret:
        print("Error: Failed to capture image.")
        break

    cv2.imshow('Live Camera Feed', frame)

    # Press 'q' to exit the camera view
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()
