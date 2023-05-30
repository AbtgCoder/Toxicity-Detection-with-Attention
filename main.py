from data_processing import prepare_dataset
from model_training import train_model
from model_evaluation import evaluate_model

import os
import pandas as pd
import tensorflow as tf


# Enable memory growth for GPUs
gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Load the dataset
dataset_path = os.path.join("jigsaw-toxic-comment-classification-challenge", "train.csv", "train.csv")
df = pd.read_csv(dataset_path)

# Preprocess the data
train_dataset, validation_dataset, test_dataset= prepare_dataset(df)

# Build and Train the model
model = train_model(train_dataset, validation_dataset)

# Evaluate the model
evaluate_model(model, test_dataset)
