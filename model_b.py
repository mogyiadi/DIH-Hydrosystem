import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_NUM_INTEROP_THREADS"] = "16"
os.environ["TF_NUM_INTRAOP_THREADS"] = "256"

import tensorflow as tf
tf.config.threading.set_inter_op_parallelism_threads(16)
tf.config.threading.set_intra_op_parallelism_threads(256)
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0
import matplotlib.pyplot as plt

data = r"house_plant_species"
batch_size = 128


train_ds = keras.utils.image_dataset_from_directory(
    data,
    validation_split=0.2,
    subset="training",
    seed=69,
    image_size=(224, 224),
    batch_size=batch_size,
    label_mode="categorical"
)

val_ds = keras.utils.image_dataset_from_directory(
    data,
    validation_split=0.2,
    subset="validation",
    seed=69,
    image_size=(224, 224),
    batch_size=batch_size,
    label_mode="categorical"
)

class_names = train_ds.class_names
print('Class names:', class_names)

with open("class_names.txt", "w") as f:
    f.write("\n".join(class_names))

# Some augmenatation
augment = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomBrightness(0.1),
], name="augmentation")

autotune = tf.data.AUTOTUNE
train_ds = (train_ds.map(lambda x, y: (augment(x, training=True), y), num_parallel_calls=autotune).prefetch(autotune))
val_ds = val_ds.prefetch(autotune)

base_model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(224, 224, 3), pooling="avg")
# Freeze the base model for the head training
base_model.trainable = False
# base_model.summary()

inputs = keras.Input(shape=(224, 224, 3))

x = base_model(inputs, training=False)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(len(class_names), activation="softmax")(x)

model = keras.Model(inputs, outputs)
model.summary()

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=float(0.001)),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks = [
    keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True),
    keras.callbacks.ModelCheckpoint('best_head.keras', save_best_only=True, save_weights_only=True),
]

print('Fine tuning the head of the model')
history = model.fit(train_ds, validation_data=val_ds, epochs=10, callbacks=callbacks)

base_model.trainable = True
for layer in base_model.layers[:200]:
    layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=float(0.0001)),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks_finetune = [
    keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True),
    keras.callbacks.ModelCheckpoint('best_finetune.keras', save_best_only=True, save_weights_only=True),
    keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3),
]

print('Fine tuning the top layers of the model')
history_finetune = model.fit(train_ds, validation_data=val_ds, epochs=20, callbacks=callbacks_finetune)

n_head = len(history.history["accuracy"])
acc = history.history["accuracy"] + history_finetune.history["accuracy"]
val = history.history["val_accuracy"] + history_finetune.history["val_accuracy"]
plt.figure()
plt.plot(acc, label="train acc")
plt.plot(val, label="val acc")
plt.axvline(n_head, color="gray", linestyle="--", label="fine-tune start")
plt.legend(); plt.title("Accuracy"); plt.savefig("training_curve.png"); plt.close()
print("Saved training_curve.png")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('model_b.tflite', 'wb') as f:
    f.write(tflite_model)
print("Saved model_b.tflite")




























