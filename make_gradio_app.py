from data_processing import preprocess_data

import os
import pandas as pd
import numpy as np
import gradio as gr
import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K

df = pd.read_csv(
    os.path.join(
        "jigsaw-toxic-comment-classification-challenge", "train.csv", "train.csv"
    )
)
X, y, vectorizer = preprocess_data(df)

# Load the Trained Model
model = tf.keras.models.load_model("att_toxicity_model.h5")


def visualize_attention(comment):
    vectorized_comment = vectorizer([comment])

    results = model.predict(vectorized_comment)
    text = ""
    for idx, col in enumerate(df.columns[2:]):
        text += f"{col}: {results[0][idx] > 0.5}\n"

    # Get the attention weights
    attention_layer = model.layers[6] # activation layer after applying softmax to attention weights
    get_attention_weights = K.function(model.inputs, [attention_layer.output])
    att_weights = get_attention_weights(vectorized_comment)[0]
    
    # Flatten the attention weights
    att_weights = np.squeeze(att_weights)
    
    # Get the tokenized words from the comment
    tokens = vectorized_comment.numpy()[0]
    words = vectorizer.get_vocabulary()
    comment_words = [words[token] for token in tokens if token != 0]  # Remove padding tokens

    # Plot the attention weights
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(comment_words)), att_weights[:len(comment_words)], align='center')
    plt.xticks(range(len(comment_words)), comment_words, rotation=45, ha='right')
    plt.xlabel('Words')
    plt.ylabel('Attention Weights')
    plt.title('Attention Visualization')
    plt.tight_layout()

    plt.savefig("att_vis.png")
    
    return text, "att_vis.png"

# Create the Gradio interface
interface = gr.Interface(
    fn=visualize_attention,
    inputs=gr.inputs.Textbox(lines=2, placeholder="Comment to Score", label="Input Text"),
    outputs=[gr.outputs.Textbox(label="Input Text"), gr.outputs.Image(type="filepath", label="Attention Map")],
)

interface.launch(share=True)
