import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from skopt import gp_minimize
from skopt.space import Real, Categorical, Integer

# Define the model architecture
def create_model(learning_rate, batch_size, epochs, regularization_strength):
    model = keras.Sequential([
        keras.layers.Dense(32, activation='relu', input_shape=(6,), kernel_regularizer=tf.keras.regularizers.l1(regularization_strength)),
        keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l1(regularization_strength))
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Define the training data
X_train = np.array([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]])
y_train = np.array([0, 1])

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Split the data into training and validation sets
from sklearn.model_selection import train_test_split
X_train_scaled, X_val_scaled, y_train, y_val = train_test_split(X_train_scaled, y_train, test_size=0.2, random_state=42)

# Define the hyperparameter space
space = [
    Real(0.001, 0.1, name='learning_rate'),
    Integer(32, 128, name='batch_size'),
    Integer(10, 100, name='epochs'),
    Real(0.01, 0.1, name='regularization_strength')
]

# Define the objective function
def objective(params):
    learning_rate, batch_size, epochs, regularization_strength = params
    model = create_model(learning_rate, batch_size, epochs, regularization_strength)
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, min_delta=0.001)
    history = model.fit(X_train_scaled, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val_scaled, y_val), callbacks=[early_stopping])
    loss, accuracy = model.evaluate(X_val_scaled, y_val)
    return -accuracy

# Perform Bayesian optimization
res_gp = gp_minimize(objective, space, n_calls=50, random_state=42)

# Get the optimal hyperparameters
optimal_params = res_gp.x

# Train the model with the optimal hyperparameters
model = create_model(*optimal_params)
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, min_delta=0.001)
history = model.fit(X_train_scaled, y_train, epochs=optimal_params[2], batch_size=optimal_params[1], validation_data=(X_val_scaled, y_val), callbacks=[early_stopping])

# Convert the model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_model = converter.convert()

# Save the TensorFlow Lite model
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

# Load the TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path='model.tflite')
interpreter.allocate_tensors()

# Get the input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Evaluate the model using the TensorFlow Lite interpreter
input_data = X_train_scaled.astype(np.float16)
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])

# Convert the output to class labels
y_pred_class = (output_data > 0.5).astype(int)

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

