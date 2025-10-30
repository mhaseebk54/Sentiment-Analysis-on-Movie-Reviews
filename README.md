# Sentiment-Analysis-on-Movie-Reviews

This project performs sentiment classification on movie reviews using a Recurrent Neural Network (LSTM) built with TensorFlow and Keras.
The dataset used is the IMDB movie reviews dataset, which contains 25,000 training and 25,000 testing samples labeled as positive or negative.

📌 Project Description

The model analyzes textual movie reviews to predict whether a review expresses a positive or negative sentiment.
It uses word embeddings and an LSTM layer to capture sequential dependencies in text data.

🧠 Model Architecture

Embedding Layer: Converts word indices into dense 32-dimensional vectors.

LSTM Layer: Processes the sequential data and learns contextual dependencies.

Dense Layer (ReLU): Adds a non-linear transformation.

Output Layer (Sigmoid): Predicts binary sentiment (0 = Negative, 1 = Positive).

🧩 Workflow

Dataset Loading:

Loaded IMDB dataset from tensorflow.keras.datasets.imdb with 10,000 most frequent words.

Split into training and testing sets: (X_train, Y_train), (X_test, Y_test).

Data Preprocessing:

Padded sequences to a fixed length (max_len = 200) using pad_sequences to ensure uniform input size.

Model Building:

Used Sequential API from Keras.

Added Embedding, LSTM, and Dense layers.

Compiled model with:

Optimizer: RMSprop

Loss: binary_crossentropy

Metric: accuracy

Training:

Trained for 5 epochs with a batch size of 128.

Used validation data (X_test, Y_test) to monitor performance.

Evaluation:

Evaluated model on test data and printed accuracy score.

🛠️ Tech Stack

Programming Language: Python

Libraries Used:

TensorFlow / Keras

NumPy

📊 Files Included

Sentiment_Analysis_Movie_Reviews.ipynb → Main Jupyter Notebook containing data loading, preprocessing, model building, training, and evaluation steps.

💡 Key Highlights

Built an LSTM-based neural network for text sentiment classification.

Used IMDB movie review dataset from Keras.

Achieved high accuracy in distinguishing positive and negative reviews.

Simple and effective example of deep learning for NLP.
