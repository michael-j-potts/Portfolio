import numpy as np
import pickle
from sklearn import linear_model
from sklearn.metrics import roc_auc_score, classification_report, roc_curve
import time

start_time = time.time()
vocabsize_icd = 5056

vocabsize = vocabsize_icd
input_seqs_icd = np.array(pickle.load(open('./patient_visit_icds', 'rb')), dtype = object)
labels = np.array(pickle.load(open('./labels', 'rb')), dtype = float)


def combine_encounter(seqs, length):
	ret_vector = np.zeros(length)
	for enc in seqs:
		for code in enc:
			ret_vector[int(code)] = 1
	return ret_vector


input_seqs = np.array([combine_encounter(input_seqs_icd[i], vocabsize_icd) for i in range(0, len(input_seqs_icd))])

trainratio = 0.7
validratio = 0.1
testratio = 0.2
trainlindex = int(len(input_seqs_icd)*trainratio)
validlindex = int(len(input_seqs_icd)*(trainratio + validratio))


print("starting logistic regression training: ")
best_aucrocs = []
for run in range(10):

	perm = np.random.permutation(input_seqs.shape[0])
	rinput_seqs = input_seqs[perm]
	rlabels = labels[perm]
	r_input_icd = input_seqs_icd[perm]

	train_input_seqs = rinput_seqs[:trainlindex]
	train_labels = rlabels[:trainlindex]

	valid_input_seqs = rinput_seqs[trainlindex:validlindex]
	valid_labels = rlabels[trainlindex:validlindex]

	test_input_seqs = rinput_seqs[validlindex:]
	test_labels = rlabels[validlindex:]
	test_input_seqs_interpretations = r_input_icd[validlindex:]

	model = linear_model.LogisticRegression().fit(train_input_seqs, train_labels)

	vpredict_probabilities = np.array([a[1] for a in model.predict_proba(valid_input_seqs)])
	print("Validation AUC_ROC: ", roc_auc_score(valid_labels, vpredict_probabilities))

	predict_probabilities = np.array([a[1] for a in model.predict_proba(test_input_seqs)])
	print("Test AUC_ROC: ", roc_auc_score(test_labels, predict_probabilities))

	best_aucrocs.append(roc_auc_score(test_labels, predict_probabilities))

print("Average AUCROC:", np.mean(best_aucrocs), "+/-", np.std(best_aucrocs))
minutes = int(time.time() - start_time)
seconds = int(((time.time() - start_time) - minutes) * 60)
print("Logistic regression completed in : ", int(minutes/60), " minutes and ", seconds, "seconds.")
