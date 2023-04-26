# %%
import tensorflow as tf
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# %%
def convert_folder_images(folder_name):

    # specify the folder path
    folder_path = f"alphabet_data/{folder_name}"

    # initialize an empty list to store the image arrays
    image_arrays = []

    # loop through each file in the folder
    for filename in os.listdir(folder_path):
        # check that the file is an image file (e.g. JPEG, PNG)
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # construct the full file path
            file_path = os.path.join(folder_path, filename)
            # read the image file as a numpy array
            img_array = cv2.imread(file_path)

            #gray the images
            # img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY) # from 3 channels to 1 channel
            # # coverting to gray scale gets rid of one dimension, so we add it back in, this time with 1 channel
            # # Add an extra dimension to the grayscale image
            # img_array = np.expand_dims(img_array, axis=-1)

            # add the image array to the list
            image_arrays.append(img_array)

    # convert the list of image arrays to a numpy array
    image_arrays = np.array(image_arrays)
    return image_arrays

# %%
def get_all_images_and_labels():
    a = convert_folder_images("a")
    # make labels for all photos based on how many images are in the folder
    # the folder names gives you the label for each folder of images

    b = convert_folder_images("b")
    c = convert_folder_images("c")
    d = convert_folder_images("d")
    e = convert_folder_images("e")

    labels_a = np.full(len(a),"a")
    labels_b = np.full(len(b),"b")
    labels_c = np.full(len(c),"c")
    labels_d = np.full(len(d),"d")
    labels_e = np.full(len(e),"e")
    # print(labels_e.shape)
    images = np.vstack((a,b,c,d,e))/255 # divide by 255 for normalization
    # labels = np.append(labels_a, (labels_b, labels_c, labels_d, labels_e))

    # Convert string labels to numerical labels using LabelEncoder
    label_encoder = LabelEncoder()
    # concatenate all the labels together
    labels = np.concatenate(
        (labels_a, labels_b, labels_c, labels_d, labels_e), axis=0)
    labels = label_encoder.fit_transform(labels)


    return images,labels

# %%
# make training, validation, and testing sets
images,labels = get_all_images_and_labels()
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=2)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=2)
# images[5][:][:][:].shape
# plt.imshow(one_image)

# %%
model_basic = tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters=64, kernel_size=2, padding='same', activation='relu', input_shape=(224, 224,3)),
    tf.keras.layers.MaxPooling2D(pool_size=2),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=2),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='softmax')
])

# model used for training
# layers: Convolution, Batch Normalization, MaxPool, Dense (Fully Connected Layer)
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=(224, 224, 3)),
    tf.keras.layers.MaxPooling2D(pool_size=2),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=2),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='softmax')
])

# type(model)

# %%

# configure the learning process
model.compile(loss='sparse_categorical_crossentropy',
              # use sparse cross entropy since your data is integer encoded, Npy hot-encoded (0s and 1s)
              optimizer='adam',
              metrics=['accuracy']),
# print(tf.config.list_physical_devices('GPU'))

# %%
# Check if GPU is available and set device accordingly
if tf.test.gpu_device_name():
    print('GPU found')
    device_name = tf.test.gpu_device_name()
else:
    print("No GPU found, using CPU instead")
    device_name = '/cpu:0'

# %%
# configure the learning process
model.compile(loss='sparse_categorical_crossentropy',
              # use sparse cross entropy since your data is integer encoded, NOT hot-encoded (0s and 1s)
              optimizer='adam',
              metrics=['accuracy']),
# print(tf.config.list_physical_devices('GPU'))

# %%
@tf.autograph.experimental.do_not_convert
def fit_model():
    model.fit(x_train,
              y_train,
              batch_size=50,
              epochs=5,
              validation_data=(x_valid, y_valid), verbose=1,
              callbacks=None)
# test the accuracy of model with testing ste
def test_set_eval():

    # Evaluate the model on test set
    score = model.evaluate(x_test, y_test, verbose=0)
    # Print test accuracy
    print('\n', 'Test accuracy:', score[1])
    # model.predict(x_test)
# export the model to the "Model" folder
def export_model():
    want_to_export = input("export model? ('true' or 'false'")
    if want_to_export:
        model_path = 'Model/keras_model.h5'
        tf.keras.models.save_model(model, model_path)

# %%
# Create a TensorFlow session and set it to use the specified device
with tf.device(device_name):
    # train the model
    fit_model()
    test_set_eval()

# %%
test_set_eval()

# %%
export_model()



