import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

MAX_FEATURES = 200000

def preprocess_data(df):
    X = df["comment_text"]
    y = df[df.columns[2:]].values

    # Tokenize the data
    vectorizer = TextVectorization(
        max_tokens=MAX_FEATURES, output_sequence_length=1800, output_mode="int"
    )
    vectorizer.adapt(X.values)

    return X, y, vectorizer

def prepare_dataset(df):
    # Preprocess the data
    X, y, vectorizer = preprocess_data(df)

    # Create the TensorFlow Data Pipeline
    vectorized_text = vectorizer(X.values)
    dataset = tf.data.Dataset.from_tensor_slices((vectorized_text, y))
    dataset = dataset.cache()
    dataset = dataset.shuffle(160000)
    dataset = dataset.batch(16)
    dataset = dataset.prefetch(8)

    # Split the dataset into train, validation, and test sets
    train_dataset = dataset.take(int(len(dataset) * 0.7))
    validation_dataset = dataset.skip(int(len(dataset) * 0.7)).take(int(len(dataset) * 0.2))
    test_dataset = dataset.skip(int(len(dataset) * 0.9)).take(int(len(dataset) * 0.1))

    return train_dataset, validation_dataset, test_dataset