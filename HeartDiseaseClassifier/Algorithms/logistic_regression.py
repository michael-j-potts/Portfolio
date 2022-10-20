import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import *
from sklearn.model_selection import train_test_split
import time

def classification_metrics(y_pred, y_true):
	accuracy = accuracy_score(y_pred, y_true)
	precision = precision_score(y_pred, y_true)
	recall = recall_score(y_pred, y_true)
	f1score = f1_score(y_pred, y_true)

	return accuracy, precision, recall, f1score

def metrics(prediction, y_test):
	acc, precision, recall, f1score = classification_metrics(prediction, y_test)
	aucroc = roc_auc_score(y_test, prediction)

	#print("Accuracy: ", acc)
	#print("Precision: ", precision)
	#print("Recall: ", recall)
	#print("F1score: ", f1score)
	#print("Aucroc: ", aucroc)

	return acc, precision, recall, f1score, aucroc

start_time = time.time()

#We begin by importing our data, setting our labels, and preparing our x value set
input_seqs = pd.read_csv('../Cleaned_Dataset/Cleaned_Data.csv', index_col = 0)
y = input_seqs.heart_dis
x = input_seqs.drop(columns = ['heart_dis'], axis = 0)

print("starting logistic regression training: ")
model = linear_model.LogisticRegression(max_iter = 3000)

accs = []
precisions = []
recalls = []
f1scores = []
aucrocs = []

for run in range(20):
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25)
	print("Epoch: ", run + 1)

	model.fit(x_train, y_train)
	prediction = model.predict(x_test)
	acc, precision, recall, f1score, aucroc = metrics(prediction, y_test)

	accs.append(acc)
	precisions.append(precision)
	recalls.append(recall)
	f1scores.append(f1score)
	aucrocs.append(aucroc)

print("Average Accuracy:", np.round(np.mean(accs), 4), "+/-", np.round(np.std(accs), 4))
print("Average Precision:", np.round(np.mean(precisions), 4), "+/-", np.round(np.std(precisions), 4))
print("Average Recall:", np.round(np.mean(recalls), 4), "+/-", np.round(np.std(recalls), 4))
print("Average F1score:", np.round(np.mean(f1scores), 4), "+/-", np.round(np.std(f1scores), 4))
print("Average AUCROC:", np.round(np.mean(aucrocs), 4), "+/-", np.round(np.std(aucrocs), 4))

minutes = int(time.time() - start_time)
seconds = int(((time.time() - start_time) - minutes) * 60)
print("Logistic regression completed in : ", int(minutes/60), " minutes and ", seconds, "seconds.")
