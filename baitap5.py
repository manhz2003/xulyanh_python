import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import cv2

# Đường dẫn đến thư mục chứa hình ảnh
data_dir = os.path.expanduser('~/Downloads/anh100')

def extract_features(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (64, 64))
    img = img.flatten()
    return img

def prepare_data(data_dir):
    features = []
    labels = []

    for image_name in os.listdir(data_dir):
        image_path = os.path.join(data_dir, image_name)
        if image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            features.append(extract_features(image_path))
            # Gán nhãn bằng cách lấy tên tệp mà không có phần mở rộng
            label = image_name.split('.')[0]  # Giả sử tên tệp là số nguyên
            labels.append(label)
        else:
            print(f"Đã bỏ qua tệp không phải hình ảnh: {image_name}")

    return np.array(features), np.array(labels)

# Chuẩn bị dữ liệu
X, y = prepare_data(data_dir)

if X.shape[0] == 0:
    print("Không tìm thấy hình ảnh trong thư mục được chỉ định.")
else:
    print(f"Loaded {X.shape[0]} images.")

# Chia dữ liệu thành tập huấn luyện và kiểm tra
if X.shape[0] > 0:  # Chỉ chia dữ liệu nếu có mẫu
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Chuyển đổi nhãn thành số nguyên và đảm bảo các giá trị nằm trong phạm vi hợp lệ
    unique_labels = np.unique(y)
    label_to_int = {label: idx for idx, label in enumerate(unique_labels)}
    y_train = np.array([label_to_int[label] for label in y_train])
    y_test = np.array([label_to_int[label] for label in y_test])

    # Naive Bayes
    nb_classifier = GaussianNB()
    nb_classifier.fit(X_train, y_train)
    y_pred_nb = nb_classifier.predict(X_test)
    print(f'Độ chính xác của Naive Bayes: {accuracy_score(y_test, y_pred_nb)}')

    # CART (Gini Index)
    cart_classifier = DecisionTreeClassifier(criterion='gini')
    cart_classifier.fit(X_train, y_train)
    y_pred_cart = cart_classifier.predict(X_test)
    print(f'CART accuracy: {accuracy_score(y_test, y_pred_cart)}')

    # ID3 (Information Gain)
    id3_classifier = DecisionTreeClassifier(criterion='entropy')
    id3_classifier.fit(X_train, y_train)
    y_pred_id3 = id3_classifier.predict(X_test)
    print(f'ID3 accuracy: {accuracy_score(y_test, y_pred_id3)}')

    # Neural Network
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=(X.shape[1],)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(len(unique_labels), activation='softmax'))  # Sửa số lớp đầu ra

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

    # Đánh giá mô hình
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print(f'Độ chính xác của mạng thần kinh: {test_acc}')
