import cv2
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (860, 620))
    cv2.imshow('QR code', frame)
    # Thoát vòng lặp nếu nhấn phím 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.imwrite("template.jpg", frame)
        break
cv2.destroyAllWindows()