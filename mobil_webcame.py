#### Connect With Mobile WebCame
import cv2
# Replace with the URL shown by IP Webcam (with /video at end)
url = 'http://192.168.1.4:8080/video'

cap = cv2.VideoCapture(url)

while True:
    success, frame = cap.read()
    if not success:
        print("Failed to grab frame.")
        break

    cv2.imshow("Mobile Camera", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
