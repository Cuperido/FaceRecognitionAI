"""
Facial Gender Recognition 
Young You (cuperido@uw.edu)

This code is developed and executed in the following environment.
- Windows 11 64-bit operating system
- PyCharm 2023.1.2 (https://www.jetbrains.com/pycharm)
- Conda 23.3.1 (https://docs.conda.io/en/main/miniconda.html)
- Python 3.10.10 (Included in Conda)
- Tensorflow 2.10 (https://www.tensorflow.org/install/pip)
- CUDA Toolkit 12.1.1 (https://developer.nvidia.com/cuda-toolkit-archive)
- cuDNN v8.9.0 (https://developer.nvidia.com/rdp/cudnn-archive)

Image source references:
- Kaggle Human Faces by Ashwin Gupta (https://www.kaggle.com/datasets/ashwingupta3012/human-faces)
- Generated Media Inc (https://generated.photos)
"""

import tensorflow as tf
from keras import layers
from keras.models import Sequential
from tensorflow.keras import regularizers

import pathlib
import matplotlib.pyplot as plt


def test_gpu(detail):
    """
    Evaluate if the system use GPU or not.
    """
    gpu = len(tf.config.list_physical_devices("GPU"))
    print("Num GPUs :", gpu)
    if gpu < 0:
        print("Warning! This system doesn't use GPU.")
    else:
        print("This system uses GPU.")

    # For a detailed evaluation
    if detail:
        tf.debugging.set_log_device_placement(True)
        a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        c = tf.matmul(a, b)
        print(c)


def load_dataset(params):
    """
    Load dataset from the given path.
    The data classes should be separated by different paths.
    """
    data_path = pathlib.Path(params["DatasetPath"])

    image_count = len(list(data_path.glob("*/*")))
    print(image_count, "file loaded.")

    datasets = []
    classes = None

    # Load dataset
    for dataset_name in params["Datasets"]:
        dataset = tf.keras.utils.image_dataset_from_directory(
            data_path,
            validation_split=params["TestSplitRatio"],
            subset=dataset_name,
            seed=params["RandomSeed"],
            shuffle=True,
            image_size=(params["ImageWidth"], params["ImageHeight"]),
            batch_size=params["BatchSize"],
        )

        # Get classes
        if classes is None:
            classes = dataset.class_names

        # Shuffle and cache data
        dataset = dataset.cache().shuffle(params["ShuffleCount"]).prefetch(buffer_size=params["CacheBufferSize"])

        # Insert to the dataset array
        datasets.append(dataset)

    datasets.append(classes)
    return datasets


def build_model_v1(params):
    """
    This model is the same network as UWImg
    """
    model = Sequential([
        layers.Rescaling(1 / 255, input_shape=(params["ImageWidth"], params["ImageHeight"], params["ImageChannel"])),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(len(params["Classes"])),
        layers.Activation("softmax"),
    ])

    optimizer = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9, decay=0.01)
    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=["accuracy"])

    return optimizer, model


def build_model_v2(params):
    """
    Adjust hyperparameters from the network of v1
    """
    model = Sequential([
        layers.Rescaling(1 / 255, input_shape=(params["ImageWidth"], params["ImageHeight"], params["ImageChannel"])),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(len(params["Classes"])),
        layers.Activation("softmax"),
    ])

    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, decay=0.001)
    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=["accuracy"])

    return optimizer, model


def build_model_v3(params):
    """
    Use the adam optimizer
    """
    model = Sequential([
        layers.Rescaling(1 / 255, input_shape=(params["ImageWidth"], params["ImageHeight"], params["ImageChannel"])),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(len(params["Classes"])),
        layers.Activation("softmax"),
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, decay=0.0001)
    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=["accuracy"])

    return optimizer, model


def build_model_v4(params):
    """
    Adjust network and hyperparameters of he v3 adam optimizer
    """
    model = Sequential([
        layers.Rescaling(1 / 255, input_shape=(params["ImageWidth"], params["ImageHeight"], params["ImageChannel"])),
        layers.Conv2D(64, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(128, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(256, kernel_size=(2, 2), strides=(1, 1), padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(len(params["Classes"])),
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, decay=0.0001)
    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=["accuracy"])

    return optimizer, model


def build_model_v5(params):
    """
    Try to solve overfitting. Increase dense and use decay stepping, data augmentation, and weight regularizers.
    """
    model = Sequential([
        layers.Rescaling(1 / 255, input_shape=(params["ImageWidth"], params["ImageHeight"], params["ImageChannel"])),
        layers.RandomFlip('horizontal'),
        layers.RandomRotation(0.1),
        layers.Conv2D(128, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(256, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(512, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.Dropout(0.3),
        layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.Dropout(0.3),
        layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.Dropout(0.3),
        layers.Dense(len(params["Classes"])),
    ])

    schedule = tf.keras.optimizers.schedules.InverseTimeDecay(initial_learning_rate=0.0001,
                                                              decay_steps=params["Epochs"] * 10,
                                                              decay_rate=0.0001,
                                                              staircase=False)

    optimizer = tf.keras.optimizers.Adam(schedule)
    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=["accuracy"])

    return optimizer, model


def evaluate_model(model, training_dataset, test_dataset, params):
    results = model.fit(
        training_dataset,
        validation_data=test_dataset,
        epochs=params["Epochs"],
    )

    return results


def draw_plot(results, params):
    plt.figure(figsize=(16, 8))
    plt.plot(range(0, params["Epochs"]), results.history["accuracy"], label="Training")
    plt.plot(range(0, params["Epochs"]), results.history["val_accuracy"], label="Validation")
    plt.legend(loc="lower right")
    plt.title('Training and Validation Accuracy')
    plt.show()


def main():
    """
    Main method of the final project
    It dispatches all questions and saves its results.
    """
    # Initialize parameters & hyperparameters.
    params = {
        "DatasetPath": "data/v2",
        "ImageWidth": 128,
        "ImageHeight": 128,
        "ImageChannel": 3,
        "BatchSize": 32,
        "RandomSeed": 0,
        "TestSplitRatio": 0.2,
        "Datasets": ["training", "validation"],
        "ShuffleCount": 100,
        "CacheBufferSize": tf.data.AUTOTUNE,
        "Epochs": 50,
    }

    # Initialize the data depot class
    print("- Evaluate GPU Status.")
    test_gpu(False)

    print("- Load image files.")
    training_dataset, test_dataset, classes = load_dataset(params)
    params["Classes"] = classes

    # Build and compile a model.
    # Only need to execute one of the model functions (build_model_v1, build_model_v2, build_model_v3...)
    print("- Build and Compile the model.")
    optimizer, model = build_model_v5(params)
    print(optimizer.get_config())
    model.summary()

    # Evaluate the model
    print("- Evaluate the model.")
    results = evaluate_model(model, training_dataset, test_dataset, params)

    # Draw a plot
    print("- Draw a plot.")
    draw_plot(results, params)

    print("- Done.")


if __name__ == '__main__':
    main()
