import time

from typing import Tuple

import numpy as np
import tflite_runtime.interpreter as tflite


def load_model_interpreter(model_path: str) -> tflite.Interpreter:
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    return interpreter


def get_embeddings(interpreter: tflite.Interpreter, image: np.array) -> Tuple[
        np.array, np.array, float]:
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # check the type of the input tensor
    floating_model = input_details[0]['dtype'] == np.float32

    # add N dim
    input_data = np.expand_dims(image, axis=0)
    input_mean = 0
    input_std = 1

    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    interpreter.set_tensor(input_details[0]['index'], input_data)

    start_time = time.time()
    interpreter.invoke()
    stop_time = time.time()

    dt = (stop_time - start_time) * 1000
    print('time: {:.3f}ms'.format(dt))

    output_data = interpreter.get_tensor(output_details[0]['index'])

    results = np.squeeze(output_data)
    top_k = results.argsort()[-5:][::-1]

    return results, top_k, dt
