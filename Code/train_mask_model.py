"""
This program loads the image dataset and uses a pre-trained
MobileNetV2 which is a lightweight CNN useful to deploy models
on mobile devices with less computing power

Model is evaluated and saved
"""

# all necessary imports
import warnings
import matplotlib.pyplot as plt
import numpy as np
import os
from imutils import paths
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
from tensorflow.keras.models import load_model

# ignore warnings
warnings.simplefilter('ignore')

# hyperparameters
LR = 1e-4
epochs = 25
batch_size = 64

# dataset path
imagePaths = list(paths.list_images("C:\\Users\\rahul\\PycharmProjects\\Universe\\dataset"))
# lists containing images with their corresponding labels
images = []
labels = []

# image path for images without mask
imagepaths_without_mask = list(paths.list_images('dataset\\without_mask'))
# image path for images with mask
imagepaths_with_mask = list(paths.list_images('dataset\\with_mask'))


# load images and labels for both classes and perform resizing and pre-processing
def load_images_labels(imagepath, given_label):
    for ip in imagepath:
        image = load_img(ip, target_size=(200, 200))
        image = img_to_array(image)
        image = preprocess_input(image)
        label = given_label
        images.append(image)
        labels.append(label)


# method to plot accuracy of the model
def plot_accuracy(history):
    # accuracy plot
    plt.figure()
    # train accuracy plot
    plt.plot([i for i in range(1, epochs + 1)], history.history["acc"], label="train_acc")
    # validation accuracy plot
    plt.plot([i for i in range(1, epochs + 1)], history.history["val_acc"], label="val_acc")
    # plot
    plt.title("Accuracy Vs Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()


# method to plot loss of the model
def plot_loss(history):
    # loss plot
    plt.figure()
    # train loss plot
    plt.plot([i for i in range(1, epochs + 1)], history.history["loss"], label="train_loss")
    # validation loss plot
    plt.plot([i for i in range(1, epochs + 1)], history.history["val_loss"], label="val_loss")
    # plot
    plt.title("Loss Vs Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


# load images and labels
load_images_labels(imagepaths_without_mask, "without_mask")
load_images_labels(imagepaths_with_mask, "with_mask")

# convert images and labels to numpy arrays
images = np.array(images, dtype="float32")
labels = np.array(labels)

# one hot encoding labels and converting to categorical values
le = LabelEncoder()
labels = le.fit_transform(labels)
labels = to_categorical(labels)

# split into train and test with 80:20 ratio
train_X, test_X, y_train, y_test = train_test_split(images, labels, test_size=0.20, random_state=42)

# image data generator to perform random transformation on train images
train_idg = ImageDataGenerator(rotation_range=40,
                               width_shift_range=0.15,
                               height_shift_range=0.15,
                               shear_range=0.1,
                               zoom_range=0.2,
                               horizontal_flip=True,
                               fill_mode='nearest')


# method to train model and save model
def train_model():
    # pre-trained MobileNetV2 with imagenet weights
    model = MobileNetV2(weights="imagenet", include_top=False,
                        input_tensor=Input(shape=(200, 200, 3)))
    # as pre-trained weights are used, do not train the layers
    for layer in model.layers:
        layer.trainable = False

    # for output of model
    head = model.output
    head = AveragePooling2D(pool_size=(7, 7))(head)
    head = Flatten(name="flatten")(head)
    head = Dense(128, activation="relu")(head)
    head = Dropout(0.3)(head)
    head = Dense(2, activation="softmax")(head)

    # create model from inputs and outputs
    model_full = Model(inputs=model.input, outputs=head)

    # compile model using adam optimizer
    lr_decay = LR / epochs
    opt = Adam(lr=LR, decay=lr_decay)
    # loss is binary crossentropy as there are 2 classes
    model_full.compile(loss="binary_crossentropy", optimizer=opt,
                       metrics=["accuracy"])
    # step size for fitting the model
    step = len(train_X) // batch_size
    val_step = len(test_X) // batch_size

    history = model_full.fit(
        train_idg.flow(train_X, y_train, batch_size=batch_size),
        steps_per_epoch=step,
        validation_data=(test_X, y_test),
        validation_steps=val_step,
        epochs=epochs)

    # make predictions on test set
    predictions = model_full.predict(test_X, batch_size=batch_size)
    predictions = np.argmax(predictions, axis=1)
    print(classification_report(y_test.argmax(axis=1), predictions,
                                target_names=le.classes_))
    # save model
    model_full.save("model_MNV2", save_format="h5")
    return history


# train model only if it does not exists
if os.path.exists("model_MNV2") is False:
    history = train_model()
    plot_accuracy(history)
    plot_loss(history)
else:
    model = load_model("model_MNV2")
    plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    model.summary()