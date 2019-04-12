import sys
import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import hashlib
sys.path.append(os.path.abspath("C:\\tmp"))
from reddit_id import *
# import nltk
# nltk.download()

from nltk.stem.snowball import SnowballStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit



reddit = get_reddit()  # gets my personal reddit info, praw
_subreddits = ['legaladvice', 'politics']  # subreddits to test
stemmer = SnowballStemmer("english", ignore_stopwords=True)  # stemming, tricky business


def sew(_subreddit, MAX_POSTS):
	postlist = []

	submission_num = 0  # counts the submission
	for submission in reddit.subreddit(_subreddit).stream.submissions():
		submission_num += 1
		if submission_num > MAX_POSTS:
			break
		print "Submission #", submission_num
		submission.comments.replace_more(limit=None)
		for comment in submission.comments.list():
			post = [comment.body, _subreddit]
			postlist.append(post)

	print "Posts: ", len(postlist)
	return postlist


def reap(postlist, name='temp'):
	df = pd.DataFrame(postlist, columns=['BODY', 'SUBREDDIT'])
	df.to_csv(name + '.csv', index=False, mode='w', encoding="utf-8")


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
	n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
	"""
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
	plt.figure()
	plt.title(title)
	if ylim is not None:
		plt.ylim(*ylim)
	plt.xlabel("Training examples")
	plt.ylabel("Score")
	train_sizes, train_scores, test_scores = learning_curve(
		estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
	train_scores_mean = np.mean(train_scores, axis=1)
	train_scores_std = np.std(train_scores, axis=1)
	test_scores_mean = np.mean(test_scores, axis=1)
	test_scores_std = np.std(test_scores, axis=1)
	plt.grid()

	plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
		train_scores_mean + train_scores_std, alpha=0.1,
		color="r")
	plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
		test_scores_mean + test_scores_std, alpha=0.1, color="g")
	plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
		label="Training score")
	plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
		label="Cross-validation score")

	plt.legend(loc="best")
	return plt


def test_nb(csv_name='temp.csv'):
	data = pd.read_csv(csv_name)  # inhales data from the CSV into the panda data frame
	numpy_array = data.values  # converts to numpy array

	X = numpy_array[:, 0]
	Y = numpy_array[:, 1]

	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=42)
	text_clf = Pipeline([
		('vect', CountVectorizer(stop_words='english')),
		('tfidf', TfidfTransformer()),
		('clf', MultinomialNB()),
	])

	text_clf = text_clf.fit(X_train, Y_train)
	predicted = text_clf.predict(X_test)

	# plotting
	title = "Learning Curves (Naive Bayes) - Doc"
	# Cross validation with 100 iterations to get smoother mean test and train
	# score curves, each time with 20% data randomly selected as a validation set.
	cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
	estimator = text_clf
	plot_learning_curve(estimator, title, X, Y, cv=cv, n_jobs=4)
	# end plotting

	return np.mean(predicted == Y_test)


def test_svm(stemmed=0, plot=0, csv_name='temp.csv'):
	# text in column 1, classifier in column 2.
	data = pd.read_csv(csv_name)
	numpy_array = data.values

	X = numpy_array[:,0]
	Y = numpy_array[:,1]

	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=42)

	class StemmedCountVectorizer(CountVectorizer):
		def build_analyzer(self):
			analyzer = super(StemmedCountVectorizer, self).build_analyzer()
			return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])

	if stemmed:
		stemmed_count_vect = StemmedCountVectorizer(stop_words='english')
	else:
		stemmed_count_vect = CountVectorizer(stop_words='english')

	text_clf_svm = Pipeline([
		('vect', stemmed_count_vect),
		('tfidf', TfidfTransformer()),
		('clf-svm', SGDClassifier(
			loss='hinge', penalty='l2'
			, alpha=1e-3
			, max_iter=30
			, tol=1e-3
			, random_state=42))
	])

	text_clf_svm = text_clf_svm.fit(X_train, Y_train)
	predicted = text_clf_svm.predict(X_test)
	# plotting
	if plot == 1:
		title = "Learning Curves - Doc Classification (SVM)"
		# Cross validation with 100 iterations to get smoother mean test and train
		# score curves, each time with 20% data randomly selected as a validation set.
		cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
		estimator = text_clf_svm
		plot_learning_curve(estimator, title, X, Y, cv=cv, n_jobs=4)
	# end plotting
	return np.mean(predicted == Y_test)


def test_forest(stemmed=0, plot=0, csv_name='temp.csv'):
	data = pd.read_csv(csv_name)
	numpy_array = data.values

	X = numpy_array[:, 0]
	Y = numpy_array[:, 1]
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=42)

	class StemmedCountVectorizer(CountVectorizer):
		def build_analyzer(self):
			analyzer = super(StemmedCountVectorizer, self).build_analyzer()
			return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])

	if stemmed == 0:
		stemmed_count_vect = StemmedCountVectorizer(stop_words='english')
	elif stemmed == 1:
		stemmed_count_vect = CountVectorizer(stop_words='english')
	else:
		stemmed_count_vect = CountVectorizer()

	text_clf_rf = Pipeline([
		('vect', stemmed_count_vect),
		('tfidf', TfidfTransformer()),
		# ('clf-rf', RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0))
		('clf-rf', RandomForestClassifier(bootstrap=True, class_weight=None, criterion='entropy',
			max_depth=None, max_features='auto', max_leaf_nodes=3,
			min_impurity_decrease=0.0, min_impurity_split=None,
			min_samples_leaf=1, min_samples_split=2,
			min_weight_fraction_leaf=0, n_estimators=2, n_jobs=None,
			oob_score=False, random_state=0, verbose=0, warm_start=True))
	])

	text_clf_rf = text_clf_rf.fit(X_train, Y_train)
	predicted = text_clf_rf.predict(X_test)
	# plotting
	if plot == 1:
		title = "Learning Curves - Doc Classification (Random Forest)"
		# Cross validation with 100 iterations to get smoother mean test and train
		# score curves, each time with 20% data randomly selected as a validation set.
		cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
		estimator = text_clf_rf
		plot_learning_curve(estimator, title, X, Y, cv=cv, n_jobs=4)
	# end plotting
	return np.mean(predicted == Y_test)


def save_reddit(posts, subreddits, name='temp'):
	postlist = []
	for subreddit in subreddits:
		print "Downloading submissions for ", subreddit
		postlist = postlist + sew(subreddit, posts)
		print "Concatenated postlist now contains ", len(postlist), " posts"
	reap(postlist, name)


save_reddit(100, ['sports', 'waterniggas', 'politics'], '3x_subreddits')
# print "Accuracy of Naive Bayes: %0.5f" % (test_nb())
# print "Accuracy of SVM - no stop: %0.5f" % (test_svm(-1))
# print "Accuracy of SVM - no stemming, full stop: %0.5f" % (test_svm(0))
print "Accuracy of SVM - stemming: %0.5f" % (test_svm(1, 1, '3x_subreddits.csv'))
# print "Accuracy of Random Forest, no stop: %0.5f" % (test_forest(-1))
# print "Accuracy of Random Forest, no stemming, full stop: %0.5f" % (test_forest(0))
print "Accuracy of Random Forest - stemming: %0.5f" % (test_forest(1, 1, '3x_subreddits.csv'))
plt.show()
"""
digits = load_digits()
X, y = digits.data, digits.target


title = "Learning Curves (Naive Bayes)"
# Cross validation with 100 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.
cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

estimator = GaussianNB()
plot_learning_curve(estimator, title, X, y, ylim=(0.7, 1.01), cv=cv, n_jobs=4)

title = r"Learning Curves (SVM, RBF kernel, $\gamma=0.001$)"
# SVC is more expensive so we do a lower number of CV iterations:
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
estimator = SVC(gamma=0.001)
plot_learning_curve(estimator, title, X, y, (0.7, 1.01), cv=cv, n_jobs=4)

plt.show()

"""