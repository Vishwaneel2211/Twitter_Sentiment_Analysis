Twitter Sentiment Analysis (PySpark + Logistic Regression)

A PySpark ML pipeline that classifies tweets as positive or negative using TF-IDF features and Logistic Regression.

Overview

This project loads a raw tweet dataset (in the Sentiment140 format), cleans and tokenizes the text, converts it into numerical features using TF-IDF, and trains a Logistic Regression classifier to predict sentiment.

Features
Loads and labels raw tweet data (no header CSV)
Converts sentiment labels to binary (0 = negative, 1 = positive)
Text cleaning (lowercasing, removing punctuation/special characters)
Tokenization and stopword removal
Feature extraction using HashingTF + IDF
Sentiment classification with LogisticRegression
Model evaluation using accuracy metric
Tech Stack
Apache Spark (PySpark)
Spark MLlib (pyspark.ml)
Dataset

