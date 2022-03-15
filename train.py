import tensorflow as tf
import tensorflow_addons as tfa


def read_data(file_path):
    with open(file_path) as f:
        lines = f.readlines()
        lines = map(lambda x: x.split(), lines)
        return list(lines)


@tf.function
def load_img(path_label):
    path = path_label[0]
    label = path_label[1]

    img = tf.io.read_file(path)
    img = tf.io.decode_image(img, channels=3, expand_animations=False)
    # return img, tf.strings.to_number(label, tf.int32)
    return img, tf.one_hot(tf.strings.to_number(label, tf.int32), 3)


def create_dataset(file_path, size=256, batch_size=16, repeat=10,
                   training=True, shuffle=True):
    # classes: 0 - vehicles, 1 - plants, 2 - others

    data = read_data(file_path)
    ds = tf.data.Dataset.from_tensor_slices(data)
    ds = ds.map(load_img, num_parallel_calls=tf.data.AUTOTUNE)

    resize_and_rescale = tf.keras.Sequential([
        tf.keras.layers.Resizing(size, size),
        tf.keras.layers.Rescaling(1./127.5, offset=-1)
    ])
    ds = ds.map(lambda x, y: (resize_and_rescale(x), y), 
                num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.cache()
    ds = ds.repeat(repeat)
    ds = ds.batch(batch_size)

    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip('horizontal_and_vertical'),
        tf.keras.layers.RandomRotation(0.2),
    ])

    ds = ds.map(lambda x, y: (augmentation(x, training=training), y),
                num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(buffer_size=1000)

    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds


class_names = ['vehicles', 'plants', 'others']
train_ds = create_dataset(
    'train_data.txt',
    size=256,
    batch_size=16,
    repeat=10,
    training=True
)
val_ds = create_dataset(
    'test_data.txt',
    size=256,
    batch_size=16,
    repeat=3,
    training=True,
    shuffle=False
)

base_model = tf.keras.applications.efficientnet_v2.EfficientNetV2S(
    include_top=False, include_preprocessing=False
)
base_model.trainable = False
layer_1x1 = tf.keras.layers.Conv2D(2048, 1)
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
prediction_layer = tf.keras.layers.Dense(len(class_names))
softmax_layer = tf.keras.layers.Softmax()

inputs = tf.keras.Input(shape=(256, 256, 3))
x = base_model(inputs, training=False)
x = layer_1x1(x)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = softmax_layer(prediction_layer(x))
model = tf.keras.Model(inputs, outputs)

metrics = [
    'accuracy',
    tfa.metrics.F1Score(num_classes=3),
    # tf.keras.metrics.Precision(class_id=0, name='vehicles_pres'),
    # tf.keras.metrics.Precision(class_id=1, name='plants_pres'),
    # tf.keras.metrics.Precision(class_id=2, name='others_pres'),
]

model.compile(
    optimizer='adam',
    loss=tf.losses.CategoricalCrossentropy(),
    metrics=metrics)

class_weight = {0: 5.0, 1: 5.0, 2: 0.5} # 150/150/1200

csv_logger = tf.keras.callbacks.CSVLogger('training.log')
saver = tf.keras.callbacks.ModelCheckpoint(
    filepath='model_{epoch}.h5',
    verbose=1,
)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=50,
    class_weight=class_weight,
    callbacks=[csv_logger, saver]
)
