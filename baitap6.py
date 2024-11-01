import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Đường dẫn đến thư mục chứa dữ liệu IRIS
data_dir = os.path.expanduser('~/Downloads/iris')
data_file = os.path.join(data_dir, 'iris.data')

# Hàm đọc dữ liệu từ file iris.data
def load_iris_data(data_file):
    features = []
    labels = []

    with open(data_file, 'r') as f:
        for line in f:
            if line.strip():  # Bỏ qua dòng trống
                parts = line.strip().split(',')
                features.append([float(x) for x in parts[:-1]])
                labels.append(parts[-1])  # Nhãn là tên loài

    return np.array(features), np.array(labels)

# Chuẩn bị dữ liệu
X, y = load_iris_data(data_file)

# Chia dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Chuyển đổi nhãn thành số nguyên
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
print(f'Độ chính xác của CART: {accuracy_score(y_test, y_pred_cart)}')

# ID3 (Information Gain)
id3_classifier = DecisionTreeClassifier(criterion='entropy')
id3_classifier.fit(X_train, y_train)
y_pred_id3 = id3_classifier.predict(X_test)
print(f'Độ chính xác của ID3: {accuracy_score(y_test, y_pred_id3)}')

# Neural Network
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(X.shape[1],)))
model.add(Dense(64, activation='relu'))
model.add(Dense(len(unique_labels), activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

# Đánh giá mô hình
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f'Độ chính xác của mạng thần kinh: {test_acc}')
