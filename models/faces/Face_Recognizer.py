import numpy as np
import tensorflow as tf


class FaceRecognizer(object):
    def __init__(self):
        print('Loading Face Recognizer...')
        self.interpreter = tf.lite.Interpreter(model_path='models/faces/Face_Recognizer.tflite')
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        print('Successfully Loaded Face Recognizer!')

    def __call__(self, landmark_list, ):
        input_details_tensor_index = self.input_details[0]['index']
        self.interpreter.set_tensor(
            input_details_tensor_index,
            np.array([landmark_list], dtype=np.float32))
        self.interpreter.invoke()

        output_details_tensor_index = self.output_details[0]['index']

        result = self.interpreter.get_tensor(output_details_tensor_index)
        result_index = np.argmax(np.squeeze(result))

        result_probability_list = list(np.squeeze(result))
        result_probability = result_probability_list[result_index]

        result_list = [f'{result_probability:.2f}', result_index]

        return result_list
