import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Configure paths
dataset = 'face_landmarks.csv'
model_output_path = 'Face_Recognizer.hdf5'

# Define constants
NUM_CLASSES = 3
RANDOM_SEED = 10
NUM_LANDMARKS = 468

# Load the data
x_dataset = np.loadtxt(dataset, delimiter=',', dtype='float32', usecols=list(range(1, (NUM_LANDMARKS * 2) + 1)))
y_dataset = np.loadtxt(dataset, delimiter=',', dtype='int32', usecols=0)  # class_num columns

# Train test split
x_train, x_test, y_train, y_test = train_test_split(x_dataset, y_dataset, train_size=0.75, random_state=RANDOM_SEED)

# Setup model architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.InputLayer((NUM_LANDMARKS * 2, )),
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
epochs = 1000
history = model.fit(
    x_train,
    y_train,
    epochs=epochs,
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

open('Face_Recognizer.tflite', 'wb').write(tflite_model)

# Configure interpreter
interpreter = tf.lite.Interpreter(model_path='Face_Recognizer.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

interpreter.set_tensor(input_details[0]['index'], np.array([x_test[0]]))
interpreter.invoke()
tflite_results = interpreter.get_tensor(output_details[0]['index'])

print(np.squeeze(tflite_results))
print(np.argmax(np.squeeze(tflite_results)))

# Plot training results
print("Calculating the accuracy...")
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
print("Calculating the loss...")
loss = history.history['loss']
val_loss = history.history['val_loss']

# Get early stopped epoch value
epochs_range = range(early_stopping_callback.stopped_epoch + 1)
print("The results are being visualized...")
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.subplot(1, 2, 2)

plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
