import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, classification_report

# Load and preprocess data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize and reshape for FFNN
x_train_ffnn = x_train / 255.0
x_test_ffnn = x_test / 255.0

# Normalize and reshape for CNN
x_train_cnn = x_train.reshape(-1, 28, 28, 1) / 255.0
x_test_cnn = x_test.reshape(-1, 28, 28, 1) / 255.0

# One-hot encode labels
y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

# FFNN model
ffnn = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])
ffnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
ffnn.fit(x_train_ffnn, y_train_cat, epochs=5, batch_size=32, verbose=0)
ffnn_score = ffnn.evaluate(x_test_ffnn, y_test_cat, verbose=0)
print(f"[FFNN] Test Accuracy: {ffnn_score[1]:.4f}")

# CNN model
cnn = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])
cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
cnn.fit(x_train_cnn, y_train_cat, epochs=5, batch_size=32, verbose=0)
cnn_score = cnn.evaluate(x_test_cnn, y_test_cat, verbose=0)
print(f"[CNN] Test Accuracy: {cnn_score[1]:.4f}")

# Plot accuracy comparison
plt.bar(['FFNN', 'CNN'], [ffnn_score[1] * 100, cnn_score[1] * 100], color=['blue', 'green'])
plt.title('Test Accuracy Comparison')
plt.ylabel('Accuracy (%)')
plt.show()

# Predictions
y_pred_ffnn = np.argmax(ffnn.predict(x_test_ffnn), axis=1)
y_pred_cnn = np.argmax(cnn.predict(x_test_cnn), axis=1)

# Confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.heatmap(confusion_matrix(y_test, y_pred_ffnn), annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title('FFNN Confusion Matrix')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')

sns.heatmap(confusion_matrix(y_test, y_pred_cnn), annot=True, fmt='d', cmap='Greens', ax=axes[1])
axes[1].set_title('CNN Confusion Matrix')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')

plt.tight_layout()
plt.show()
