This folder contains scripts for several programs:


### LearningLatentRepresentationsProject (completed):
This project is a summary of my final work for my class CS 598 Deep Learning for Health Care. This project worked to evaluate, update and extend the works of the given authors in classifying patient survival outcomes for ICU stays using the MIMIC-iii dataset. My final results show a minor improvement in performance over the original authors through refinement of feature selection. This is accomplished through
feature reduction using autoencoders to create a latent space model, then fed into various models including logistic regression, and various LSTM models.

### PredictivePediatricClassification (skeleton):
This project worked by converting Iquvia claims data into organized patient data based on ICD-10 Code. This was part of my internship, but include some permitted algorithmic models. The goal of this project was to classify pediatric patients illness (RSV positive) as well as thier likelihood of hospitalization using Iquvia data (as a start), and the N3C database down the road. During this time, I worked on and became very familiar with Neural Backed Decision Trees. Please feel free to read more about Neural Backed Decision Trees here:
https://arxiv.org/abs/2004.00221
https://research.alvinwan.com/neural-backed-decision-trees/

I will also be uploading a working version of my implementation of Neural Backed Decision Trees soon, but my work is currently incomplete.

### StockPredictionModel (in progress, but phase 1 complete):
I have included a working copy of my stock prediction program, which converts daily stock information into a time series data set, and uses several neural network models to predict stock prices.

To be completed: creating a text mining webcrawler to evaluate daily new releases about a specific stock to determine if the stock is a strong, average or weak buy/sell, or neutral. These features will be fed into the phase 1 model to further enrich performance.