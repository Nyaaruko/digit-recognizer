from tkinter import *
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from PIL import ImageGrab

# Tải mô hình đã lưu
model = tf.keras.models.load_model('model_neuron_digit_recog.keras')

# Tạo cửa sổ chính
root = Tk()
root.title("Digit Recognizer")
canvas = Canvas(root, width=640, height=640, bg='black')
canvas.pack()

# Khởi tạo các biến để vẽ
last_x, last_y = None, None

def draw(event):
    global last_x, last_y
    x, y = event.x, event.y
    if last_x is not None and last_y is not None:
        canvas.create_line((last_x, last_y, x, y), fill='white', width=10)
    last_x, last_y = x, y

def reset(event):
    global last_x, last_y
    last_x, last_y = None, None

canvas.bind("<B1-Motion>", draw)  # Vẽ khi giữ chuột trái
canvas.bind("<ButtonRelease-1>", reset)  # Reset khi thả chuột

def predict_digit():
    # Lưu ảnh từ canvas vào mảng
    x = root.winfo_rootx() + canvas.winfo_x() + 20
    y = root.winfo_rooty() + canvas.winfo_y() + 20
    x1 = x + canvas.winfo_width()
    y1 = y + canvas.winfo_height()

    # Chụp ảnh từ canvas
    img_canva = ImageGrab.grab().crop((x, y, x1, y1))  # Lấy ảnh từ canvas
    img = img_canva.convert('L')  # Chuyển đổi thành ảnh grayscale
    img = img.resize((28, 28))  # Thay đổi kích thước về 28x28

    # Đọc file ảnh
    img = img_to_array(img)  # Chuyển đổi thành mảng numpy
    img = img.astype('int')  # Chuẩn hóa
    img = np.reshape(img, (1, 28, 28, 1))  # Thay đổi kích thước để phù hợp với đầu vào mô hình
    
    # Dự đoán lớp
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions, axis=1)[0]
    
    # Hiển thị kết quả
    plt.imshow(img.reshape(28, 28), cmap='gray')
    plt.axis('off')
    plt.title(f'Predicted: {predicted_class}')
    plt.show()

def clear_canvas():
    canvas.delete("all")  # Xóa tất cả các đối tượng trên canvas

# Tạo frame để chứa các nút
button_frame = Frame(root)
button_frame.pack(pady=10)

# Tạo nút dự đoán
predict_button = Button(button_frame, text="Predict", command=predict_digit)
predict_button.pack(side=LEFT, padx=5)

# Tạo nút xóa
clear_button = Button(button_frame, text="Clear", command=clear_canvas)
clear_button.pack(side=LEFT, padx=5)

# Khởi động vòng lặp chính
root.mainloop()
