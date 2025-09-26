# ReviewGlow
Explanation of Sentiment Analysis on IMDb Dataset
This script performs sentiment analysis on movie reviews from the IMDb dataset, classifying them as positive or negative using two approaches: a traditional machine learning (ML) model with scikit-learn and a deep learning (DL) model with TensorFlow. It also includes functionality to predict sentiment for custom text inputs. Below is a step-by-step explanation of each component.
1. Purpose and Overview
The goal is to classify movie reviews as positive (1) or negative (0) based on their text content. The IMDb dataset, containing 50,000 reviews (25,000 for training, 25,000 for testing), is used. The script compares two methods:

Traditional ML: Logistic Regression with TF-IDF vectorization, which transforms text into numerical features based on word importance.
Deep Learning: A Recurrent Neural Network (RNN) with Long Short-Term Memory (LSTM) layers, which processes text as sequences to capture context.
The script also allows users to input custom text to predict sentiment using the trained DL model.

2. Prerequisites
To run the script, you need:

Python: Version 3.8 or higher for compatibility with libraries.
Libraries:

NumPy for numerical operations and array handling.
TensorFlow for building and training the RNN model.
TensorFlow Datasets to load the IMDb dataset.
Scikit-learn for TF-IDF vectorization, logistic regression, and accuracy evaluation.


Internet: Required to download the IMDb dataset.
Hardware: A CPU is sufficient, but a GPU accelerates training the deep learning model.

3. Loading the IMDb Dataset
The IMDb dataset is loaded using TensorFlow Datasets, which provides a convenient way to access pre-labeled movie reviews. The dataset is split into training (25,000 reviews) and test (25,000 reviews) sets. Each review is a text string, and each label is a binary value (0 for negative, 1 for positive). The dataset is loaded in a supervised format, providing pairs of text and labels, along with metadata (e.g., dataset description). The text data is initially in tensor format, which is converted to strings, and labels are converted to integers for compatibility with both scikit-learn and TensorFlow.
4. Data Preprocessing
The raw text data must be preprocessed differently for the ML and DL models:

For scikit-learn (ML):

The text reviews are stored as lists of strings.
Labels are converted to NumPy arrays, as scikit-learn expects numerical arrays for training and evaluation.


For TensorFlow (DL):

The same text and label lists are used but undergo further processing (described later) to create sequences suitable for the RNN model.
This step ensures the data is in the correct format for each model’s requirements.



5. Traditional Machine Learning: Logistic Regression with TF-IDF
This section builds a logistic regression model using TF-IDF features for sentiment classification.
TF-IDF Vectorization

Purpose: Converts text reviews into numerical features by measuring word importance.
Process:

A TF-IDF vectorizer is used to transform text into a sparse matrix of features.
It limits the vocabulary to the top 10,000 words to reduce dimensionality and computational cost.
English stop words (e.g., "the", "is") are removed to focus on meaningful words.
The vectorizer is fitted on the training data to learn the vocabulary and term weights, then applied to both training and test data to ensure consistent feature representation.


TF-IDF Concept: TF-IDF (Term Frequency-Inverse Document Frequency) assigns higher weights to words that are frequent in a document but rare across the dataset, highlighting distinctive terms.

Logistic Regression

Purpose: Trains a classifier to predict sentiment based on TF-IDF features.
Process:

A logistic regression model is initialized with a maximum of 200 iterations to ensure convergence during training.
The model is trained on the TF-IDF features of the training data and their corresponding labels.
Predictions are made on the test data’s TF-IDF features.


Evaluation: The accuracy is calculated by comparing predicted labels to true test labels, providing a measure of how well the model generalizes.

6. Deep Learning: RNN with LSTM
This section builds an RNN model with LSTM layers to classify sentiment by processing text as sequences.
Text Preprocessing for RNN

Tokenization:

A tokenizer converts each review into a sequence of integers, where each integer represents a word in a vocabulary of the top 10,000 words.
An out-of-vocabulary token (<OOV>) is used for words not in the vocabulary.
The tokenizer is fitted on the training data to build the vocabulary, then applied to both training and test data.


Padding:

Sequences vary in length (due to differing review lengths), so they are padded or truncated to a fixed length of 100 tokens.
Padding adds zeros at the end of shorter sequences, and truncation removes tokens from the end of longer sequences to ensure uniform input size for the RNN.



RNN Model Architecture

Purpose: Builds a neural network to capture sequential patterns in text for sentiment classification.
Layers:

Embedding Layer: Converts each word’s integer index into a dense 64-dimensional vector, allowing the model to learn word representations.
LSTM Layer: A 64-unit LSTM processes the sequence of word embeddings, capturing long-term dependencies and context. It outputs a single vector (not a sequence) for classification.
Dense Layer (ReLU): A 32-unit layer with ReLU activation adds non-linearity to combine LSTM outputs.
Dense Layer (Sigmoid): A single-unit layer with sigmoid activation outputs a probability (0 to 1) for positive sentiment.


Why LSTM?: LSTMs are effective for text because they handle sequential data and mitigate vanishing gradient issues, remembering important context over long sequences.

Training and Evaluation

Compilation:

The model uses binary cross-entropy loss, suitable for binary classification.
The Adam optimizer is used for efficient gradient-based optimization.
Accuracy is tracked as a metric during training.


Training:

The model is trained for 5 epochs (iterations over the dataset) with a batch size of 32 reviews.
20% of the training data is reserved for validation to monitor performance on unseen data during training.


Evaluation:

The model is evaluated on the test set, computing both loss and accuracy.
The test accuracy is reported, indicating how well the model generalizes.



7. Predicting Sentiment on Custom Text

Purpose: Allows users to input new text and predict its sentiment using the trained RNN model.
Process:

A function takes a text string, tokenizes it using the trained tokenizer, and pads it to the same length (100 tokens) used during training.
The RNN model predicts a probability (0 to 1). A threshold of 0.5 determines the sentiment: above 0.5 is positive, below is negative.
The function returns “Positive” or “Negative” for user-friendly output.


Example Inputs:

A positive review (e.g., “This movie was absolutely fantastic and thrilling!”) is expected to be classified as positive.
A negative review (e.g., “Terrible plot and boring acting.”) is expected to be classified as negative.



8. Expected Results

Logistic Regression Accuracy: Typically ranges from 85% to 90%, as TF-IDF with logistic regression is effective for text classification and often outperforms simple deep learning models on this task.
LSTM Accuracy: Typically ranges from 80% to 87% after 5 epochs. The RNN may underperform compared to logistic regression due to the simple architecture, limited training epochs, or lack of hyperparameter tuning.
Custom Predictions: The model correctly identifies clear positive and negative sentiments in sample texts, though performance may vary for ambiguous or short inputs.
