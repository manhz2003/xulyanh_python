import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import resize
import matplotlib.pyplot as plt
from collections import Counter

# Đường dẫn tới thư mục ảnh
data_dir = os.path.expanduser('~/Downloads/anh100')

# Hàm để tải và xử lý ảnh
def load_images(data_dir, img_size=(64, 64)):
    features = []
    labels = []

    print(f"Checking directory: {data_dir}")  # In ra đường dẫn của thư mục chính
    print("Files in the directory:")

    # Lặp qua tất cả các tệp trong thư mục
    for file in os.listdir(data_dir):
        img_path = os.path.join(data_dir, file)
        if file.endswith(('.jpg', '.jpeg', '.png')):  # Kiểm tra phần mở rộng tệp
            print(f"Reading image: {img_path}")  # In ra đường dẫn hình ảnh
            img = imread(img_path)

            # Kiểm tra số chiều của hình ảnh
            if img.ndim == 2:  # Hình ảnh đã là grayscale
                img_gray = img
            elif img.ndim == 3:  # Hình ảnh RGB
                img_gray = rgb2gray(img)
            else:
                print(f"Unexpected image shape: {img.shape}")
                continue  # Bỏ qua hình ảnh nếu không phải dạng 2D hoặc 3D

            img_resized = resize(img_gray, img_size).flatten()  # Chuyển đổi thành vector 1D
            features.append(img_resized)

            # Tách nhãn từ tên tệp (giả sử tên tệp có định dạng "nhan.jpg")
            label = os.path.splitext(file)[0]  # Lấy tên tệp mà không có phần mở rộng
            labels.append(label)  # Sử dụng tên tệp làm nhãn

    print(f"Total images loaded: {len(features)}")  # In ra tổng số hình ảnh đã tải
    return np.array(features), np.array(labels)

# Tải dữ liệu ảnh
features, labels = load_images(data_dir)

# In ra nhãn và số lượng mỗi lớp
print("Labels:", set(labels))
print("Label counts:", Counter(labels))

# Danh sách các tỷ lệ chia train-test
splits = [(0.8, 0.2), (0.7, 0.3), (0.6, 0.4), (0.4, 0.6)]

# Huấn luyện và đánh giá cho mỗi tỷ lệ chia
for train_size, test_size in splits:
    X_train, X_test, y_train, y_test = train_test_split(features, labels, train_size=train_size, test_size=test_size, random_state=42)

    # Kiểm tra kích thước tập dữ liệu
    print(f"Training set size: {len(y_train)}, Test set size: {len(y_test)}")

    # KNN
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)
    knn_accuracy = accuracy_score(y_test, y_pred_knn)
    print(f"KNN Accuracy with train-test split {train_size*100}-{test_size*100}: {knn_accuracy:.2f}")

    # SVM
    svm = SVC(kernel='linear', C=1.0)
    svm.fit(X_train, y_train)
    y_pred_svm = svm.predict(X_test)
    svm_accuracy = accuracy_score(y_test, y_pred_svm)
    print(f"SVM Accuracy with train-test split {train_size*100}-{test_size*100}: {svm_accuracy:.2f}")
