import cv2

cap = cv2.VideoCapture(0)  # 0 = default webcam

if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

while True:
    success, frame = cap.read()
    if not success:
        print("Error: Could not read frame")
        break

    frame = cv2.flip(frame, 1)  # mirror it so it feels natural, like a mirror

    cv2.imshow("Webcam Feed", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()