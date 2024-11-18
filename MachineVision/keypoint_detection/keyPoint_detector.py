import os
import tensorflow as tf
import json
import numpy as np
import matplotlib.pyplot as plt
import cv2

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, Reshape, Dropout, Flatten, Dense
from tensorflow.keras.applications import ResNet152V2

def load_image(img):
    byte_image = tf.io.read_file(img)
    image = tf.io.decode_jpeg(byte_image)
    image = tf.reshape(image, shape=image_shape)
    return image

# def load_labels(label_path):
#     with open(label_path.numpy(), 'r', encoding='utf-8') as f:
#         label = json.load(f)
#         pick_point_coords = label['shapes'][0]['points'][0]
#         coords = np.array([pick_point_coords[0], pick_point_coords[1]], dtype=np.float32)
#     return tf.convert_to_tensor(coords, dtype=tf.float32)

# def set_label_shape(x):
#     x.set_shape([2])  # Explicitly set the shape to (2,) for the x, y coordinates
#     return x

def load_labels(label_path):
    ### [bottle_bounding-box], [cap_bounding-box], [pick_point]
    with open(label_path, 'r', encoding='utf-8') as f:
        label = f.readlines()
        print(label)

current_script = os.path.dirname(__file__)
image_shape = (512, 612, 3)
label_path = current_script+'/txt_labels/1.txt'
load_labels(label_path)





'''
######## LOAD TRAIN DATA ########
#################################
train_images_path = current_script + '/data/images/train/*.jpg'
train_labels_path = current_script + '/data/labels/train/*.json'


train_images = tf.data.Dataset.list_files(train_images_path, shuffle=False)
### in order to get the real images:
train_images = train_images.map(load_image)
# train_images = train_images.map(lambda x: tf.image.resize(x, (250, 250)))
# train_images = train_images.map(lambda x: x/255)

train_labels = tf.data.Dataset.list_files(train_labels_path, shuffle=False)
train_labels = train_labels.map(lambda x: tf.py_function(load_labels, [x], tf.float32))
train_labels = train_labels.map(set_label_shape)
######## LOAD TEST DATA ########
#################################
test_image_path = current_script + '/data/images/test/*.jpg'
test_labels_path = current_script + '/data/labels/test/*.json'

test_images = tf.data.Dataset.list_files(test_image_path, shuffle=False)
test_images = test_images.map(load_image)
# test_images = test_images.map(lambda x: tf.image.resize(x, (250, 250)))
# test_images = test_images.map(lambda x: x/255)

test_labels = tf.data.Dataset.list_files(test_labels_path, shuffle=False)
test_labels = test_labels.map(lambda x: tf.py_function(load_labels, [x], tf.float32))
test_labels = test_labels.map(set_label_shape)

######## LOAD VALIDATION DATA ########
######################################
val_image_path = current_script + '/data/images/val/*.jpg'
val_labels_path = current_script + '/data/labels/val/*.json'

val_images = tf.data.Dataset.list_files(val_image_path, shuffle=False)
val_images = val_images.map(load_image)
# val_images = val_images.map(lambda x: tf.image.resize(x, (250, 250)))
# val_images = val_images.map(lambda x: x/255)

val_labels = tf.data.Dataset.list_files(val_labels_path, shuffle=False)
val_labels = val_labels.map(lambda x: tf.py_function(load_labels, [x], tf.float32))
val_labels = val_labels.map(set_label_shape)

######## COMBINE IMAGE WITH LABELS ########
###########################################
train = tf.data.Dataset.zip((train_images, train_labels))
# train = train.shuffle(3)
train = train.batch(4)
train = train.prefetch(4)  ### load on the memory as train it

test = tf.data.Dataset.zip((test_images, test_labels))
# test = test.shuffle(3)
test = test.batch(4)
test = test.prefetch(4)  ### load on the memory as train it

val = tf.data.Dataset.zip((val_images, val_labels))
# val = val.shuffle(1)
val = val.batch(4)
val = val.prefetch(4)  ### load on the memory as train it


######## SHOW SOME SAMPLE ########
##################################
data_sample = train.as_numpy_iterator()
res = data_sample.next()
print((res[1]).shape)

fig, ax = plt.subplots(ncols=4, figsize=(20,20))
for idx in range(4): 
    sample_image = res[0][idx]  # Convert to numpy array
    sample_image = np.array(sample_image, dtype=np.uint8)
    # x, y = int(res[1][0][idx]), int(res[1][1][idx])
    # print(x, y)

    # cv2.circle(sample_image, (x, y), 10, (255,0,0), -1)
    ax[idx].imshow(sample_image)
# plt.show()

######## MODEL ########
#######################
model = Sequential([
    Input(shape=image_shape),
    ResNet152V2(include_top=False, input_shape=image_shape), ### resnet is a pretrained neuralnetwork
    Conv2D(512, 3, padding='same', activation='relu'),
    Conv2D(512, 3, padding='same', activation='relu'),
    Conv2D(256, 3, 2, padding='same', activation='relu'), ### a filter of 3 pixel by 3 pixels which moves 2 pixels 2 pixles at a time
    Conv2D(256, 2, 2, activation='relu'),
    Dropout(0.05),
    Flatten(),
    Dense(128, activation='relu'),  # Add a Dense layer to process the flattened features
    Dense(2, activation='linear'),  # Output 2 points (x, y)
    ]) 
print(model.summary())

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
loss = tf.keras.losses.MeanSquaredError()
model.compile(optimizer, loss)

for image_batch, label_batch in train.take(1):
    print("Image batch shape:", image_batch.shape)  # Expected: (batch_size, 512, 612, 3)
    print("Label batch shape:", label_batch.shape)
print((train))
hist = model.fit(train, epochs=20, validation_data=val)
# print(model_history.history)

# plt.plot(model_history.history['loss'], color='teal', label='loss')
# plt.plot(model_history.history['val_loss'], color='orange', label='val loss')
# plt.suptitle('Loss')
# plt.legend()
# plt.show()
'''