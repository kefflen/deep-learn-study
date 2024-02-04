from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
#from keras.layers.normalization import BatchNormalization
import numpy as np
from keras.preprocessing import image

filterNumbers = 64
featureMapsDimension = (3,3)
imageDimensionsAndColorCanal = (64, 64, 3)

classifier = Sequential([
    Conv2D(filterNumbers, featureMapsDimension, input_shape=imageDimensionsAndColorCanal, activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    #BatchNormalization(),
    Conv2D(filterNumbers, featureMapsDimension, input_shape=imageDimensionsAndColorCanal, activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(units=128, activation = 'relu'),
    Dropout(0.2),
    Dense(units=128, activation = 'relu'),
    Dropout(0.2),
    Dense(units=1, activation = 'sigmoid')
])

classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

gen_train = ImageDataGenerator(rescale=1./255,
                               rotation_range=7,
                               horizontal_flip=True,
                               shear_range=0.2,
                               height_shift_range=0.07,
                               zoom_range=0.2)

gen_test = ImageDataGenerator(rescale=1./255)
train_base = gen_train.flow_from_directory('datasets/dogs-and-cats/training_set',
                                           target_size=(64, 64),
                                           batch_size=32,
                                           class_mode='binary')

test_base = gen_test.flow_from_directory('datasets/dogs-and-cats/test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')

classifier.fit(train_base, steps_per_epoch=4000/32, epochs=10, validation_data=test_base, validation_steps=1000/32)

# Making a prediction
imageTest = image.load_img('datasets/dogs-and-cats/test_set/gato/cat.3509.jpg', target_size=(64, 64))
imageTest = image.img_to_array(imageTest)
imageTest /= 255
imageTest = np.expand_dims(imageTest, axis=0)
result = classifier.predict(imageTest)
print(result)
print(train_base.class_indices)