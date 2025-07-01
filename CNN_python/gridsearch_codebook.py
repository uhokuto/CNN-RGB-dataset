
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn import svm
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier



def svc(X_train,y_train,X_test,y_test):
	
	#parameters = [{'kernel': ['rbf'], 'gamma': [0.001, 0.0001],'C': [1, 10, 100, 1000]}]
	parameters = [{'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
	clf = GridSearchCV(estimator=SVC(), param_grid=parameters, cv=2, iid=False)
	#学習モデルを作成
	clf.fit(X_train, y_train)
	#精度を確認
	
	best_clf = clf.best_estimator_ #ここにベストパラメータの組み合わせが入っています
	print('best parameters :',best_clf)
	print('score: {:.2%}'.format(best_clf.score(X_train,y_train)))
	y_pred = clf.predict(X_test)
	print('score: {:.2%}'.format(best_clf.score(X_test,y_test)))
	return clf
	
def random_forest(X_train,y_train,X_test,y_test):
#作成する決定木の数#決定木の深さ #分岐し終わったノードの最小サンプル数 #決定木が分岐する際に必要なサンプル数
	parameters = {'n_estimators' :[3,5,10,30,50],'max_depth' :[3,5,8,10],'min_samples_leaf': [2,5,10,20,50],'min_samples_split': [2,5,10,20,50], 'random_state' :[7,42]}

	clf = GridSearchCV(estimator=RandomForestClassifier(), param_grid=parameters, cv=2, iid=False)
	#学習モデルを作成
	clf.fit(X_train, y_train)
	#精度を確認
	best_clf = clf.best_estimator_ #ここにベストパラメータの組み合わせが入っています
	print('score: {:.2%}'.format(best_clf.score(X_train,y_train)))
	y_pred = clf.predict(X_test)
	print('score: {:.2%}'.format(best_clf.score(X_test,y_test)))
	return clf

def sgd(X_train,y_train,X_test,y_test):
	
	clf = SGDClassifier(loss='hinge')
	for i in range(1000):
		clf.partial_fit(X_train, y_train, classes=[0, 1])
		if i%200==0:
			print(np.sqrt(np.sum((clf.decision_function(X_test))**2)))
	
	print('score: {:.2%}'.format(clf.score(X_train,y_train)))
	#y_pred = clf.predict(X_test)
	print('score: {:.2%}'.format(clf.score(X_test,y_test)))
	print('loss_function_',clf.loss_function_)
	
	return clf


