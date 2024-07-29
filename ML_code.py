import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

# Define the model architecture
def create_model():
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(6,)),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

# Compile the model
def compile_model(model):
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Define the training data
X_train = np.array([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]])
y_train = np.array([0, 1])

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train the model
model = create_model()
model = compile_model(model)
model.fit(X_train_scaled, y_train, epochs=100, batch_size=32)

# Evaluate the model
y_pred = model.predict(X_train_scaled)
y_pred_class = (y_pred > 0.5).astype(int)

# Evaluate the model using various metrics
accuracy = accuracy_score(y_train, y_pred_class)
precision = precision_score(y_train, y_pred_class)
recall = recall_score(y_train, y_pred_class)
f1 = f1_score(y_train, y_pred_class)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# Print the classification report
print("Classification Report:")
print(classification_report(y_train, y_pred_class))

# Print the confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_train, y_pred_class))
