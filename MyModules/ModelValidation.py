## Defining the scoring function
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score

def print_score(clf, X_train, X_test, y_train, y_test, train=False, cv=False, threshold=False, margins=False):
	'''
	print the accuracy score, classification report and confusion matrix of classifier
	'''
	start = "\033[1;4m"
	end = "\033[0;0m"

	print(start+'Classifier'+ end +': ' + str(clf)+'\n')
		
	if train:
		'''
		training performance
		'''
		if threshold:
			y_proba = clf.predict_proba(X_train)[:, 1]
			y_pred = [1 if p >= threshold else 0 for p in y_proba]
		else:
			y_pred = clf.predict(X_train)
		
		print(start+"Train Result:"+end+"\n")
		print("\033[4mConfusion Matrix\033[0;0m:\n{}\n".format(confusion_matrix_df(clf, y_train, y_pred, margins=margins)))
		print("\033[4mAccuracy Score\033[0;0m: {0:.4f}".format(accuracy_score(y_train, y_pred)))
		print("\n\033[4mClassification Report\033[0;0m:\n{}".format(classification_report(y_train, y_pred, digits=4)))
		
		if cv:
			print('Running '+str(cv)+'-fold cross validation:')
			res = cross_val_score(clf, X_train, y_train, cv=cv, scoring='accuracy')
			print("Average Accuracy: \t{0:.4f}".format(np.mean(res)))
			print("Accuracy SD: \t\t{0:.4f}".format(np.std(res)))
		print('-----'*20)
	else:
		'''
		test performance
		'''
		if threshold:
			y_proba = clf.predict_proba(X_test)[:, 1]
			y_pred = [1 if p >= threshold else 0 for p in y_proba]
		else:
			y_pred = clf.predict(X_test)
			
		print(start+"Test Result:"+end+"\n")
		print("\033[4mConfusion Matrix\033[0;0m:\n{}\n".format(confusion_matrix_df(y_test, y_pred, margins=margins)))
		print("\033[4mAccuracy Score\033[0;0m: {0:.4f}".format(accuracy_score(y_test, y_pred)))
		print("\n\033[4mClassification Report\033[0;0m:\n{}\n".format(classification_report(y_test, y_pred, digits=4)))
		
		if cv:
			print('Running '+str(cv)+'-fold cross validation:')
			res = cross_val_score(clf, X_test, y_test, cv=cv, scoring='accuracy')
			print("Average Accuracy: \t{0:.4f}".format(np.mean(res)))
			print("Accuracy SD: \t\t{0:.4f}".format(np.std(res)))
		print('-----'*20)

def confusion_matrix_df(clf, y_true, y_pred, margins=False):
	#cm = confusion_matrix(y_true, y_pred)
	#cm_df = pd.DataFrame(cm, index=clf.classes_, columns=clf.classes_)
	#cm_df.index.name = 'True'
	#cm_df.columns.name = 'Pred >>'
	
	y_actu = pd.Series(y_true, name='Actual')
	y_pred = pd.Series(y_pred, name='Predicted')
	cm_df = pd.crosstab(y_actu, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=margins)
	return cm_df
	
	
	

	
	
## Construct ROC curve for binary classification
## Put the output in the ROC & KS Excel file and get the value of ROC, AUC, KS, Gini
def roc_curve_excel(y_true, y_score, decile='decile'):

	roc_data = pd.DataFrame(list(zip(y_true,y_score)), columns=['y_true','y_score'])
	roc_data.head()

	roc_data['threshold'] = pd.qcut(roc_data['y_score'], 10)
	roc_data['decile'] = pd.qcut(roc_data['y_score'], 10, labels=False)	
		
	return roc_data.pivot_table(index=decile, columns='y_true', aggfunc='count')
'''
USAGE:
y_train_score = grid2.decision_function(X_train)
y_train_pred = grid2.predict(X_train)
y_test_score = grid2.decision_function(X_test)
y_test_pred = grid2.predict(X_test)

ms.roc_curve_excel(y_train, y_train_score).to_csv('train_set.csv')
ms.roc_curve_excel(y_test, y_test_score).to_csv('test_set.csv')

ms.roc_curve_excel(y_test, y_test_score,decile='threshold')

'''



## Combine the y_true, y_pred, y_score
def combine_results(y_true, y_pred, y_score):
	test = pd.concat([y_true.reset_index(), pd.Series(y_pred), pd.Series(y_score)], axis=1, ignore_index=False)
	test.columns = ['index', 'y_true', 'y_pred', 'y_score']
	return test


## Combine the y_true, y_pred, y_score
def series_combiner(ndarray_list, columns=False):
	series_list = [pd.Series(a) for a in ndarray_list]
	test = pd.concat(series_list, axis=1, ignore_index=False)
	if columns:
		test.columns = columns
	return test


def variance_inflation_factor(df):
	"""
	Caculates the VIF for predictors
	"""
	vif = pd.DataFrame()
	vif["VIF Factor"] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
	vif["features"] = df.columns
	return vif

		