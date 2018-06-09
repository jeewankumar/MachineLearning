import pandas as pd
import numpy as np
import pylab as plt

#import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-white')
import sklearn, sklearn.svm

### Visualize a classifier like (logistic regression, random forest, SVM) ##########################
from matplotlib.colors import ListedColormap
def plot_binary_classifier(clf, X, y, alpha=0.2, h=0.02):   # should be 2D feature space 

	# initialize custom marker and color-map :  
	#markers = ('o', '^', 's', 'x', 'v', 'D', '*')
	markers = ('o', 'o', 'o', 'o', 'o', 'o', 'o')
	colors = ('red', 'blue', 'green', 'cyan', 'magenta', 'yellow', 'gray')
	cmap = ListedColormap(colors[:len(np.unique(y))])

	#fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(6, 5))
	
	# create a mesh to plot in
	buffer=0.5
	x_min, x_max = X[:, 0].min() - buffer, X[:, 0].max() + buffer
	y_min, y_max = X[:, 1].min() - buffer, X[:, 1].max() + buffer
	xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
	
	# Plot the decision boundary.
	Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
	Z = Z.reshape(xx.shape) # Put the result into a color plot
	plt.contourf(xx, yy, Z, alpha=alpha, cmap=cmap) #cmap = 'afmhot'

	# Plot sample points
	for idx, cls in enumerate(np.unique(y)):
		plt.scatter(x=X[y == cls, 0], y=X[y == cls, 1], c=cmap(idx), s=30,marker=markers[idx], label=cls)#,edgecolors='black')
		
	plt.legend(loc='upper right', frameon=True, framealpha=0.3, borderaxespad=0.9)
	# additionally, support vectors have a white hole in a marker  
	if isinstance(clf, sklearn.svm.classes.SVC):

		# get the separating hyperplane
		x_coordinate = np.linspace(x_min,x_max,3)
		w = clf.coef_[0]
		a = -w[0] / w[1]
		yy_hyper = a * x_coordinate - (clf.intercept_[0]) / w[1]
		plt.plot(x_coordinate, yy_hyper, 'k-')
		
		# plot the parallels to the separating hyperplane that pass through the support vectors
		support_vectors_1 = clf.support_vectors_[y[clf.support_]==1]
		b = support_vectors_1[clf.decision_function(support_vectors_1).argmax()]
		yy_down = a * x_coordinate + (b[1] - a * b[0])
		plt.plot(x_coordinate, yy_down, 'b--')
		
		support_vectors_0 = clf.support_vectors_[y[clf.support_]==0]
		b = support_vectors_0[clf.decision_function(support_vectors_0).argmin()]
		yy_up = a * x_coordinate + (b[1] - a * b[0])
		plt.plot(x_coordinate, yy_up, 'b--')
		
		## Plotting Support Vectors
		sv = clf.support_vectors_
		#plt.scatter(sv[:,0], sv[:,1], c='w', marker='.', s=25)
		plt.scatter(sv[:, 0], sv[:, 1], s=120,facecolors='none',edgecolors='black')
		plt.xlim(x_min, x_max)
		plt.ylim(y_min, y_max)				
		print('\033[1m Number of Support Vectors(Class, SV):\033[0m', list(zip(clf.classes_, clf.n_support_)) )
	
	plt.xlim(xx.min(), xx.max())
	plt.ylim(yy.min(), yy.max())


## Define a function to plot classifier
def plot_classifier(clf, X, y, alpha=0.2, h=0.02, target_names=None, hyperplane=False):
	colors = 'brkyg' ## colors for different classes
	## h=0.02 step size in the mesh
	## create a mesh to plot in
	x_min, x_max = X[:,0].min() - 1, X[:,0].max() + 1
	y_min, y_max = X[:,1].min() - 1, X[:,1].max() + 1
	xx, yy = np.meshgrid(np.arange(x_min,x_max,h), np.arange(y_min,y_max,h))

	## plot decision boundary
	Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
	#Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
	plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=alpha)
	
	## Plot the training points
	for i, color in zip(clf.classes_, colors):
		idx = np.where(y==i)
		plt.scatter(X[idx,0], X[idx,1], c=color, cmap=plt.cm.Paired, s=50, label=i) #edgecolor='k'
	## plot support vectors
	plt.scatter(clf.support_vectors_[:,0], clf.support_vectors_[:,1], c='w', marker='.', s=30)
	#plt.scatter(clf.support_vectors_[:,0], clf.support_vectors_[:,1], s=150, linewidth=1, facecolors='none', edgecolor='k')
	print('\033[1m Number of Support Vectors(Class, SV):\033[0m', list(zip(clf.classes_, clf.n_support_)) )
	
	xmin, xmax = plt.xlim()
	ymin, ymax = plt.ylim()
	if hyperplane:
		# Plot the three one-against-all clfs
		coef = clf.coef_
		intercept = clf.intercept_

		def plot_hyperplane(c, color):
			def line(x0):
				return (-(x0 * coef[c, 0]) - intercept[c]) / coef[c, 1]
			plt.plot([xmin, xmax], [line(xmin), line(xmax)], ls="--", color=color)

		if len(clf.classes_)==2:
			plot_hyperplane(0, 'k')
		elif len(clf.classes_)>2:
			for i, color in zip(clf.classes_, colors):
				plot_hyperplane(i, color)
	
	plt.xlim(xmin,xmax)
	plt.ylim(ymin,ymax)
	plt.legend()
############# GRIDSEARCH VISUALIZATION FOR GBM ###########################################

def vis_gsearch(gsearch, param_name='param_n_estimators'):
	tt = pd.DataFrame(gsearch.cv_results_)
	cols_required = ['mean_test_score']
	for i in tt.columns:
		if 'param_' in i:
			cols_required.append(i)
	tt1 = tt.loc[:,cols_required]
	cols_required.remove(param_name)
	cols_required.remove('mean_test_score')
	tt2 = tt.loc[:,cols_required].drop_duplicates().reset_index().drop('index',1)
	print(tt2)

	for row in range(tt2.shape[0]):
		temp_str = ''
		temp_str += (tt2.columns[0].replace('param_','') + " : " + str(tt2.iloc[row,0]) + "  ")
		temp = tt1.loc[tt1.loc[:,tt2.columns[0]] == tt2.loc[row,tt2.columns[0]]].reset_index().drop('index',1)
		for col in range(tt2.shape[1]-1):
			col += 1
			temp = temp.loc[temp.loc[:,tt2.columns[col]] == tt2.loc[row,tt2.columns[col]]].reset_index().drop('index',1)
			temp_str += (tt2.columns[col].replace('param_','') + " : " + str(tt2.iloc[row,col]) + "  ")
		plt.plot(temp.loc[:,param_name],temp.loc[:,'mean_test_score'],'*-',label = temp_str)
	plt.legend()
	plt.title(gsearch.best_params_)
	plt.show()

	
## Plotting ROC Curve ###############
from sklearn.metrics import roc_curve, auc
def plot_roc(y_true, y_score, buffer=0.005):
	fpr, tpr, threshold = roc_curve(y_true, y_score, drop_intermediate=False)
	roc_auc = auc(fpr, tpr)

	plt.title('Receiver Operating Characteristic')
	plt.plot(fpr, tpr, color='b', label = 'AUC = %0.4f' % roc_auc)
	plt.legend(loc = 'lower right')
	plt.plot([0, 1], [0, 1],'r--')
	plt.xlim([0-buffer, 1])
	plt.ylim([0, 1+buffer])
	plt.ylabel('True Positive Rate')
	plt.xlabel('False Positive Rate')
	plt.show()
	
	
def make_meshgrid(x, y, h=.02):
	"""Create a mesh of points to plot in

	Parameters
	----------
	x: data to base x-axis meshgrid on
	y: data to base y-axis meshgrid on
	h: stepsize for meshgrid, optional

	Returns
	-------
	xx, yy : ndarray
	"""
	x_min, x_max = x.min() - 1, x.max() + 1
	y_min, y_max = y.min() - 1, y.max() + 1
	xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
	return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
	"""Plot the decision boundaries for a clf.

	Parameters
	----------
	ax: matplotlib axes object
	clf: a clf
	xx: meshgrid ndarray
	yy: meshgrid ndarray
	params: dictionary of params to pass to contourf, optional
	"""
	Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
	Z = Z.reshape(xx.shape)
	out = ax.contourf(xx, yy, Z, **params)
	return out

	
	
def freq_table(df):
    for col in df.columns:
        if len(np.unique(df[col])) <= 10:
            print(df[col].value_counts(dropna=False))
            print('---------------------------------------')

			
import seaborn as sns
def correlation_matrix(corr, threshold=0, mask=True):
	if mask:
		mask = np.zeros_like(corr, dtype=np.bool)
		mask[np.triu_indices_from(mask)] = True
	
	plt.figure(figsize=(24,8))
	sns.heatmap(corr[corr > threshold], mask=mask, annot=True, cmap='viridis')
	plt.xticks(fontsize=12)
	plt.yticks(rotation=0, fontsize=12)
	