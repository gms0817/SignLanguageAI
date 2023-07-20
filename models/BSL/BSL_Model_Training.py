import csv
import os

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Configure paths
dataset = 'BSL_landmarks.csv'
model_output_path = 'BSL_Recognizer.hdf5'

# Define constants
NUM_CLASSES = 2
RANDOM_SEED = 10

# Load the data
x_dataset = np.loadtxt(dataset, delimiter=',', dtype='float32', usecols=list(range(1, (21 * 2) + 1)))
y_dataset = np.loadtxt(dataset, delimiter=',', dtype='int32', usecols=0)  # class_num columns

# Train test split
x_train, x_test, y_train, y_test = train_test_split(x_dataset, y_dataset, train_size=0.75, random_state=RANDOM_SEED)

# Setup model architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.InputLayer((21 * 2, )),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(20, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
])

model.summary()

# Configure callbacks
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    model_output_path, verbose=1, save_weights_only=False
)

early_stopping_callback = tf.keras.callbacks.EarlyStopping(patience=20, verbose=1)

# Compile the model
model.compile(
    optimizer='Adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
model.fit(
    x_train,
    y_train,
    epochs=1000,
    batch_size=128,
    validation_data=(x_test, y_test),
    callbacks=[checkpoint_callback, early_stopping_callback]
)

# Evaluate the model
validation_loss, validation_accuracy = model.evaluate(x_test, y_test, batch_size=128)

# Save the model
model.save(model_output_path, include_optimizer=False)

# Convert and export as tflite model
tf_converter = tf.lite.TFLiteConverter.from_keras_model(model)
tf_converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = tf_converter.convert()

open('BSL_Recognizer.tflite', 'wb').write(tflite_model)

# Configure interpreter
interpreter = tf.lite.Interpreter(model_path='BSL_Recognizer.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

interpreter.set_tensor(input_details[0]['index'], np.array([x_test[0]]))
interpreter.invoke()
tflite_results = interpreter.get_tensor(output_details[0]['index'])

print(np.squeeze(tflite_results))
print(np.argmax(np.squeeze(tflite_results)))
