import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

# Carregar os dados e dividir entre treino e teste
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalizar as imagens: `uint8` -> `float32`
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

# Garantir que as imagens tenham a forma (28, 28, 1)
x_train = tf.expand_dims(x_train, -1)
x_test = tf.expand_dims(x_test, -1)

# Converter os rótulos em vetores binários
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# Definir o modelo sequencial
model = keras.Sequential(
    [
        layers.Conv2D(32, (3, 3), input_shape=(28, 28, 1), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(10, activation="softmax"),
    ]
)

# Compilar o modelo
model.compile(
    loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
)

# Treinar o modelo
model.fit(x_train, y_train, batch_size=128, epochs=5, validation_split=0.1)

# Avaliar o modelo
model.evaluate(x_test, y_test)
