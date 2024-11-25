import time

from typing import Tuple

import numpy as np
from tensorflow import lite as tflite


class Model:
    name: str
    interpreter: tflite.Interpreter

    def __init__(self, model_path: str, model_name: str):
        self.name = model_name
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors

    def get_embeddings(self, image: np.array) -> Tuple[
            np.array, np.array, float]:
        self.interpreter.allocate_tensors()
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()

        print(f"{'='*25} Embeddings {'='*25}")
        # add N dim
        input_data = np.expand_dims(image, axis=0).astype(np.float32)
        print(input_data[0][:5])

        self.interpreter.set_tensor(input_details[0]['index'], input_data)

        start_time = time.time()
        self.interpreter.invoke()
        stop_time = time.time()

        dt = (stop_time - start_time) * 1000
        # print('time: {:.3f}ms'.format(dt))

        output_data = self.interpreter.get_tensor(output_details[0]['index'])
        print(output_data[0][:5])

        results = np.squeeze(output_data)
        top_k = results.argsort()[-5:][::-1]

        return results, top_k, dt
