import argparse
import numpy as np
import tensorflow as tf
from os import listdir, path
from PIL import Image


def classify(model_path, img_dir, threshold):
    # classes: 0 - vehicles, 1 - plants, 2 - others
    model = tf.keras.models.load_model(model_path, compile=False)
    files = listdir(img_dir)
    for file_name in files:
        file_path = path.join(img_dir, file_name)
        try:
            img = Image.open(file_path)
        except:
            continue

        img = img.resize((256, 256), Image.BILINEAR)

        inputs = np.array(img, dtype=np.float32) / 127.5 - 1.0
        if len(inputs.shape) != 3:
            continue # skip single channel image

        inputs = inputs[..., :3] # lose transparency channel
        inputs = np.expand_dims(inputs, axis=0)
        predicted = model.predict(inputs)
        predicted_class = np.argmax(predicted > threshold)

        if predicted_class == 0:
            print(f'{file_name} vehicle {predicted[0][0]}')

        if predicted_class == 1:
            print(f'{file_name} plant {predicted[0][1]}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Asset classifier')
    parser.add_argument('--model', help='H5 model', required=True)
    parser.add_argument('--dir', help='Path to images dir', required=True)
    parser.add_argument('--threshold', help='classification threshold',
                        default=0.5, type=float)
    args = parser.parse_args()

    classify(args.model, args.dir, args.threshold)
