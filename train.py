from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
import glob2

num_classes = 2
resnet_weights_path = 'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

model = Sequential()
model.add(ResNet50(include_top=False, pooling='avg', weights=resnet_weights_path))
model.add(Dense(num_classes, activation='softmax'))

model.layers[0].trainable = False

model.compile(optimizer='rmsprop', 
                     loss='binary_crossentropy', 
                     metrics=['accuracy'])

image_size = 224
batch_size = 5
train_dir = "images/train"
val_dir = "images/val"
epochs = 15

data_generator = ImageDataGenerator(preprocessing_function=preprocess_input,
                                   horizontal_flip=True,
                                   width_shift_range = 0.2,
                                   height_shift_range=0.2)

train_generator = data_generator.flow_from_directory(
                                        directory=train_dir,
                                        target_size=(image_size, image_size),
                                        color_mode='rgb',
                                        batch_size=batch_size,
                                        class_mode='binary')

val_batch_size = len(glob2.glob(f'{val_dir}/**/*.*'))

data_generator_no_aug = ImageDataGenerator(preprocessing_function=preprocess_input)
validation_generator = data_generator_no_aug.flow_from_directory(
                                        directory=val_dir,
                                        target_size=(image_size, image_size),
                                        color_mode='rgb',
                                        batch_size=val_batch_size,
                                        class_mode='binary')

train_batch_size = len(glob2.glob(f'{train_dir}/**/*.*'))
steps_per_epoch = train_batch_size / batch_size

history = model.fit_generator(train_generator,
                                       steps_per_epoch=steps_per_epoch,
                                       validation_data=validation_generator,
                                       validation_steps=len(validation_generator),
                                       epochs=epochs,
                                       verbose=1)

import matplotlib.pyplot as plt
# loss
plt.plot(history.history["loss"], label="train loss")
plt.plot(history.history["val_loss"], label="val loss")
plt.legend()
plt.show()
plt.savefig("model_output/LossVal_loss")

# accuracies
plt.plot(history.history["accuracy"], label="train acc")
plt.plot(history.history["val_accuracy"], label="val acc")
plt.legend()
plt.show()
plt.savefig("model_output/AccVal_acc")

print("Saving the model")
model.save("model_output/tiger_detector_model.h5")
