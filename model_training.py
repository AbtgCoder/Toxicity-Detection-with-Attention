from data_processing import MAX_FEATURES

import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Embedding,
    Bidirectional,
    LSTM,
    Dense,
    Activation,
    Flatten,
    RepeatVector,
    Permute,
    Multiply,
    Lambda,
)
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K


def build_model():
    input_shape = (1800, )
    inp = Input(shape=input_shape)
    emb = Embedding(input_dim=MAX_FEATURES + 1, output_dim=32)(inp)

    lstm_layer = Bidirectional(LSTM(32, return_sequences=True))(emb)

    x_a = Dense(64, kernel_initializer="glorot_uniform", activation="tanh")(lstm_layer)
    x_a = Dense(1, kernel_initializer="glorot_uniform", activation="linear")(x_a)
    x_a = Flatten()(x_a)
    att_out = Activation("softmax")(x_a)
    x_a2 = RepeatVector(64)(att_out)
    x_a2 = Permute([2, 1])(x_a2)
    mult = Multiply()([lstm_layer, x_a2])
    weighted_sum =  Lambda(lambda x : K.sum(x, axis=1), output_shape=lambda s: (s[0], s[2]))(mult)

    d1 = Dense(128, activation="relu")(weighted_sum)
    d2 = Dense(256, activation="relu")(d1)
    d3 = Dense(128, activation="relu")(d2)
    d4 = Dense(6, activation="sigmoid")(d3)

    model = Model(inputs=inp, outputs=d4)
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), optimizer="Adam")
    model.summary()

    return model


def train_model(train_dataset, validation_dataset):
    # Build the model
    model = build_model()

    # Train the model
    history = model.fit(train_dataset, epochs=5, validation_data=validation_dataset)

    # Save the trained model
    model.save("att_toxicity_model.h5")

    return model


