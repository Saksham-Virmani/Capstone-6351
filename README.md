# Enhancing Stock Price Prediction through Data Augmentation and Spatial Feature Modeling Using CNN-VAEs

## Authors
- Abishek Baskaran
- Shahni Gautam
- Saksham Virmani

## Advisor
- Prof. Yawo Kobara

## Institution
- Worldquant University

## Keywords
Deep learning, quantitative finance, data augmentation, sentiment analysis, convolutional neural networks, variational auto-encoders

## Project Overview
Accurate prediction of stock prices is crucial for making informed investment decisions. This project addresses the challenges in stock price prediction, such as limited and noisy data, and intricate dependencies, by proposing a deep learning architecture that combines Convolutional Neural Networks (CNNs) and Variational Auto-Encoders (VAEs).

### Objectives
1. **Classify Historical Data by Regime Identification:** Segment historical data into market regimes to convert the regression problem into a classification problem.
2. **Incorporate Alternative Data Sources:** Enhance data quality by integrating data from social media, economic indicators, and news.
3. **Model Spatial Features using CNNs:** Transform financial data into images using Gramian Angular Field transformations and extract spatial features with CNNs.
4. **Model Latent Space with VAEs:** Train VAEs on spatial features to learn a lower-dimensional representation, facilitating data augmentation.
5. **Generate Synthetic Data for Training:** Use VAEs to produce synthetic data, expanding the training dataset to improve model generalization.
6. **Train a Deep-Learning Predictor:** Develop a model to classify data points into market regimes based on augmented data.
7. **Back-Test with Real-World Data:** Validate the model's predictive capability by back-testing with historical data.
