
"""
Membership inference attack against a deep net classifier on the CIFAR10 dataset.
Evaluate the target model and attack model After adding gaussian noise, pepper and salt noise to dataset.
"""
import random
import numpy as np

from absl import app
from absl import flags

import tensorflow as tf
from keras import layers
# from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import time
from mia.estimators import ShadowModelBundle, AttackModelBundle, prepare_attack_data

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

NUM_CLASSES = 10
WIDTH = 32
HEIGHT = 32
CHANNELS = 3
SHADOW_DATASET_SIZE = 4000
ATTACK_TEST_DATASET_SIZE = 4000


FLAGS = flags.FLAGS
flags.DEFINE_integer(
    "target_epochs", 12, "Number of epochs to train target and shadow models."
)
flags.DEFINE_integer("num_shadows", 10, "Number of epochs to train shadow models.")
flags.DEFINE_integer("attack_epochs", 12, "Number of epochs to train attack models.")


def get_data(noiseStrength = 0):
    """Prepare CIFAR10 data."""
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
    y_train = tf.keras.utils.to_categorical(y_train)
    y_test = tf.keras.utils.to_categorical(y_test)
    X_train = X_train.astype("float32")
    X_test = X_test.astype("float32")
    y_train = y_train.astype("float32")
    y_test = y_test.astype("float32")
    # print("X_train.shape = {0}" % X_train.shape)

    # 1 add gaussian noise to the data.
    # noise1 = np.random.normal(0, noiseStrength, size = X_train.shape)
    # noise2 = np.random.normal(0, noiseStrength, size = X_test.shape)
    # X_train = X_train + noise1
    # X_test  = X_test  + noise2

    # 2 add sp noise to the data
    X_train = sp_noise(X_train, noiseStrength)
    X_test = sp_noise(X_test, noiseStrength)

    # normalize
    # X_train /= 255.0
    # X_test /= 255.0
    return (X_train, y_train), (X_test, y_test)

def sp_noise(image,prob):
    '''
    Adding salt and pepper noise to the image
    image: original image
    prob: the ratio of noise
    '''
    output = np.zeros(image.shape,np.uint8)
    noise_out = np.zeros(image.shape,np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):

            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
                noise_out[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
                noise_out[i][j] = 255
            else:
                output[i][j] = image[i][j]
                noise_out[i][j] = 100
    return output


def target_model_fn():
    """The architecture of the target (victim) model.

    The attack is white-box, hence the attacker is assumed to know this architecture too."""

    model = tf.keras.models.Sequential()

    model.add(
        layers.Conv2D(
            32,
            (3, 3),
            activation="relu",
            padding="same",
            input_shape=(WIDTH, HEIGHT, CHANNELS),
        )
    )
    model.add(layers.Conv2D(32, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))

    model.add(layers.Conv2D(64, (3, 3), activation="relu", padding="same"))
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))

    model.add(layers.Flatten())

    model.add(layers.Dense(512, activation="relu"))
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(NUM_CLASSES, activation="softmax"))
    model.compile("adam", loss="categorical_crossentropy", metrics=["accuracy"])

    return model


def attack_model_fn():
    """Attack model that takes target model predictions and predicts membership.

    Following the original paper, this attack model is specific to the class of the input.
    AttachModelBundle creates multiple instances of this model for each class.
    """
    model = tf.keras.models.Sequential()

    model.add(layers.Dense(128, activation="relu", input_shape=(NUM_CLASSES,)))

    model.add(layers.Dropout(0.3, noise_shape=None, seed=None))
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dropout(0.2, noise_shape=None, seed=None))
    model.add(layers.Dense(64, activation="relu"))

    model.add(layers.Dense(1, activation="sigmoid"))
    model.compile("adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def demo(argv):

    del argv  # Unused.

    # add different ratio of sp noise to dataset.
    for round in range(0, 1, 1):
        time_start = time.time()
        noise = round / 40
        print("round {0}, noise strength = {1}".format(round, noise))
        (X_train, y_train), (X_test, y_test) = get_data(noise)

        # Train the target model.
        print("Training the target model...")
        target_model = target_model_fn()

        target_model.fit(
            X_train, y_train, epochs=FLAGS.target_epochs, validation_split=0.1, verbose=True
        )

        # Train the shadow models.
        smb = ShadowModelBundle(
            target_model_fn,
            shadow_dataset_size=SHADOW_DATASET_SIZE,
            num_models=FLAGS.num_shadows,
        )

        # We assume that attacker's data were not seen in target's training.
        attacker_X_train, attacker_X_test, attacker_y_train, attacker_y_test = train_test_split(
            X_test, y_test, test_size=0.1
        )
        print(attacker_X_train.shape, attacker_X_test.shape)

        print("Training the shadow models...")
        X_shadow, y_shadow = smb.fit_transform(
            attacker_X_train,
            attacker_y_train,
            fit_kwargs=dict(
                epochs=FLAGS.target_epochs,
                verbose=True,
                validation_data=(attacker_X_test, attacker_y_test),
            ),
        )

        # ShadowModelBundle returns data in the format suitable for the AttackModelBundle.
        amb = AttackModelBundle(attack_model_fn, num_classes=NUM_CLASSES)

        # Fit the attack models.
        print("Training the attack models...")
        amb.fit(
            X_shadow, y_shadow, fit_kwargs=dict(epochs=FLAGS.attack_epochs, verbose=True)
        )

        # Test the success of the attack.

        # Prepare examples that were in the training, and out of the training.
        data_in = X_train[:ATTACK_TEST_DATASET_SIZE], y_train[:ATTACK_TEST_DATASET_SIZE]
        data_out = X_test[:ATTACK_TEST_DATASET_SIZE], y_test[:ATTACK_TEST_DATASET_SIZE]

        # Compile them into the expected format for the AttackModelBundle.
        attack_test_data, real_membership_labels = prepare_attack_data(
            target_model, data_in, data_out
        )

        # Compute the attack accuracy.
        attack_guesses = amb.predict(attack_test_data)
        attack_accuracy = np.mean(attack_guesses == real_membership_labels)

        # Compute the attack accuracy.
        attack_guesses = amb.predict(attack_test_data)
        attack_accuracy = np.mean(attack_guesses == real_membership_labels)

        # Compute Precision, Recall, F1-score
        precision = precision_score(real_membership_labels, attack_guesses, average='binary')
        recall = recall_score(real_membership_labels, attack_guesses, average='binary')
        f1 = f1_score(real_membership_labels, attack_guesses, average='binary')

        print('Acc: %.3f' % attack_accuracy)
        print('Precision: %.3f' % precision)
        print('Recall: %.3f' % recall)
        print('F1: %.3f' % f1)

        time_end = time.time()
        print('time cost', time_end - time_start, 's')


if __name__ == "__main__":
    app.run(demo)
