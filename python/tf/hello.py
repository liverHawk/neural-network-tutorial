import matplotlib.pyplot as plt
import os
import pandas as pd
import tensorflow as tf

from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

print(tf.__version__)

PATH = os.path.abspath("../../Iris.csv")
iris_data = pd.read_csv(PATH)

x = iris_data.drop(columns=["Id", "Species"])
y = iris_data["Species"]

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
y_onehot = tf.keras.utils.to_categorical(y_encoded)

x_train, x_test, y_train, y_test = train_test_split(x, y_onehot, test_size=0.2, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)

embedding_dim = 16

model = tf.keras.Sequential([
    layers.Dense(10, activation="relu", input_shape=(4,)),
    layers.Dropout(0.2),
    layers.Dense(8, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(3, activation="softmax"),
])

model.summary()

model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
    )

epochs = 100
history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=16,
        validation_data=(x_val, y_val),
        verbose=1,
    )

loss, accuracy = model.evaluate(x_test, y_test)

print("loss", loss)
print("Accuracy", accuracy)

history_dict = history.history
print(history_dict.keys())

acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.savefig("loss.png")
