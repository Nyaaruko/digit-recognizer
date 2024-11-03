import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from time import sleep

# Tải mô hình đã lưu
model = tf.keras.models.load_model('model_neuron_digit_recog.keras')

# Khởi tạo video capture
cap = cv2.VideoCapture(0)  # 0 is the default camera

# Đặt ngưỡng diện tích tối thiểu cho đường viền
MIN_CONTOUR_AREA = 0  # Adjust this value as needed 800
MAX_CONTOUR_AREA = 15000  # Adjust this value as needed 3000

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Chuyển đổi frame thành ảnh grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Làm mờ ảnh để giảm nhiễu
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Tạo mặt nạ cho các vùng tối
    _, dark_mask = cv2.threshold(blurred, 160, 255, cv2.THRESH_BINARY)

    # Tìm các đường viền trong vùng tối
    contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Tìm đường viền lớn nhất trong vùng tối có diện tích lớn hơn ngưỡng
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) > MIN_CONTOUR_AREA and cv2.contourArea(largest_contour) < MAX_CONTOUR_AREA:
            x, y, w, h = cv2.boundingRect(largest_contour)

            # Mở rộng vùng ROI
            padding = 20  # Điều chỉnh giá trị này để phóng to hơn hoặc nhỏ hơn
            x_start = max(x - padding, 0)
            y_start = max(y - padding, 0)
            x_end = min(x + w + padding, frame.shape[1])
            y_end = min(y + h + padding, frame.shape[0])

            digit_roi = gray[y_start:y_end, x_start:x_end]
            if digit_roi.size > 0:  # Check if the ROI is not empty
                global img28
                img28 = cv2.resize(digit_roi, (28, 28))

                # Chuyển đổi thành mảng numpy
                img = img_to_array(img28)
                img = img.astype('float32')/255.0  # Chuẩn hóa
                img = np.reshape(img, (1, 28, 28, 1))  # Thay đổi kích thước để phù hợp với đầu vào mô hình
                
                # Dự đoán lớp
                predictions = model.predict(img)
                predicted_class = np.argmax(predictions, axis=1)[0]
                
                # Vẽ khung và hiển thị kết quả trên frame
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(frame, f'Predicted: {predicted_class}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                cv2.putText(frame, f'Area: {cv2.contourArea(largest_contour)}', (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # Hiển thị frame
    cv2.imshow('Digit Recognizer - Original', frame)
    try:    
        cv2.imshow('Digit Recognizer - img28', img28)
    except:
        pass
    sleep(0.1)

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng camera khi đóng ứng dụng
cap.release()
cv2.destroyAllWindows()