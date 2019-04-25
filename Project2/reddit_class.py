import sys
import os
import json
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
	"""
	This will gather posts from reddit.
	:param _subreddit: The subreddit to harvest
	:param MAX_POSTS: The amount of posts to get
	:return: simple list of posts.
	"""
	postlist = []

	post_num = 1  # counts the submission
	for submission in reddit.subreddit(_subreddit).stream.submissions():
		submission.comments.replace_more(limit=None)
		for comment in submission.comments.list():
			if post_num > MAX_POSTS:
				break
			print "Post #", post_num
			post = [comment.body, _subreddit]
			postlist.append(post)
			post_num += 1
		if post_num > MAX_POSTS:
			break

	print "Posts: ", len(postlist)
	return postlist


def get_data(subreddit_names, max_posts):
	sub_count = 0
	comments_dict = {'comments': [], "subreddit name": []}

	for name in subreddit_names:
		for comment in reddit.subreddit(name).stream.comments():
			data = vars(comment)
			comments_dict['comments'].append(data['body'])
			comments_dict['subreddit name'].append(data['subreddit_name_prefixed'])

			sub_count += 1
			if sub_count > max_posts:
				sub_count = 0
				break

	return comments_dict


def reap(postlist, name='temp'):
	"""
	This will convert the postlist to a dataframe.
	:param postlist: the simple list returned from sew()
	:param name: The name of the file to save the list to
	:return:
	"""
	df = pd.DataFrame(postlist, columns=['BODY', 'SUBREDDIT'])
	df.to_json(name + '.json', orient='index', force_ascii=False)


def export_data(data, file_name):
	with open(file_name, "w") as export_file:
		json.dump(data, export_file, indent=2)


def read_data(file_name):
	df = pd.read_json(file_name, orient='columns')
	return df


# From : https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
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


def test_nb(data):
	"""
	Demo test, but interesting none-the-less
	:param data pandas dataframe
	:return: returns the average correctness
	"""
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


def test_svm(data, stemmed=0, plot=0, title="Learning Curves - Doc Classification (SVM)"):
	"""
	Trains and demonstrates an SVM classifier
	:param data: pandas dataframe
	:param stemmed: whether it should use nltk stemming
		1 = using nltk stemming
		0 = using standard stemming
		-1 = using standard stemming, no stop words
	:param plot:
	:return:
	"""
	# text in column 1, classifier in column 2.
	numpy_array = data.values

	X = numpy_array[:,0]
	Y = numpy_array[:,1]

	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=42)

	#  Default ignore stop words
	class StemmedCountVectorizer(CountVectorizer):
		def build_analyzer(self):
			analyzer = super(StemmedCountVectorizer, self).build_analyzer()
			return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])

	if stemmed == 0:
		stemmed_count_vect = StemmedCountVectorizer(stop_words='english')
	elif stemmed == 2:
		stemmed_count_vect = StemmedCountVectorizer()
	elif stemmed == 1:
		stemmed_count_vect = CountVectorizer(stop_words='english')
	else:
		stemmed_count_vect = CountVectorizer()

	text_clf_svm = Pipeline([
		('vect', stemmed_count_vect),
		('tfidf', TfidfTransformer()),
		('clf-svm', SGDClassifier(
			loss='hinge', penalty='l2'
			, alpha=1e-3
			, max_iter=40
			, tol=1e-3
			, random_state=42))
	])

	text_clf_svm = text_clf_svm.fit(X_train, Y_train)
	predicted = text_clf_svm.predict(X_test)
	# plotting
	if plot == 1:

		# Cross validation with 100 iterations to get smoother mean test and train
		# score curves, each time with 20% data randomly selected as a validation set.
		cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
		estimator = text_clf_svm
		plot_learning_curve(estimator, title, X, Y, cv=cv, n_jobs=4)
	# end plotting
	return np.mean(predicted == Y_test)


def test_forest(data, stemmed=0, plot=0, title="Learning Curves - Doc Classification (Random Forest)"):
	"""
	Trains and evaluates a random forest
	:param data: pandas dataframe
	:param stemmed: Whether to use stemming or stops
		0 = Uses nltk stemming with stop words
		1 = uses standard stemming with stop words
		-1 = uses standard stemming, no stop words
	:param plot:
	:return:
	"""
	numpy_array = data.values

	X = numpy_array[:, 0]
	Y = numpy_array[:, 1]
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=42)

	#  Default removes stop words
	class StemmedCountVectorizer(CountVectorizer):
		def build_analyzer(self):
			analyzer = super(StemmedCountVectorizer, self).build_analyzer()
			return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])

	if stemmed == 0:
		stemmed_count_vect = StemmedCountVectorizer(stop_words='english')
	elif stemmed == 2:
		stemmed_count_vect = StemmedCountVectorizer()
	elif stemmed == 1:
		stemmed_count_vect = CountVectorizer(stop_words='english')
	else:
		stemmed_count_vect = CountVectorizer()

	text_clf_rf = Pipeline([
		('vect', stemmed_count_vect),
		('tfidf', TfidfTransformer()),
		# ('clf-rf', RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0))
		('clf-rf', RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
			max_depth=None, max_features=None, max_leaf_nodes=10,
			min_impurity_decrease=0, min_impurity_split=None,
			min_samples_leaf=1, min_samples_split=2,
			min_weight_fraction_leaf=0, n_estimators=2, n_jobs=None,
			oob_score=False, random_state=0, verbose=0, warm_start=True))
	])

	text_clf_rf = text_clf_rf.fit(X_train, Y_train)
	predicted = text_clf_rf.predict(X_test)
	# plotting
	if plot == 1:

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
	return 0


def export_reddit(posts, subreddits, name='temp'):
	data = get_data(subreddits, posts)
	export_data(data, name)


""" THE BEATINGS WILL CONTINUE UNTIL CLASSIFICATION IMPROVES """
doc_name = '2x_json_test.json'
# export_reddit(200, ['sports', 'politics'], doc_name)
data = read_data(doc_name)

print "Accuracy of SVM - Stemming w/ stop: %0.5f" % (test_svm(data, 1, 1, "Learning Curve SVM, Stemming, Stop Words"))
print "Accuracy of SVM - Stemming w/o stop: %0.5f" % (test_svm(data, -1, 1, "Learning Curve SVM, Stemming, No Stop Words"))
print "Accuracy of SVM - nltk Stemming w/stop: %0.5f" % (test_svm(data, 0, 1, "Learning Curve SVM, NLTK Stemming, Stop Words"))
print "Accuracy of SVM - nltk Stemming w/o stop: %0.5f" % (test_svm(data, 2, 1, "Learning Curve SVM, NLTK Stemming, No Stop Words"))
print "Accuracy of Random Forest, Stemming w/ stop: %0.5f" % (test_forest(data, 1, 1, "Learning Curve Random Forest, Stemming, Stop Words"))
print "Accuracy of Random Forest, Stemming w/o stop: %0.5f" % (test_forest(data, -1, 1, "Learning Curve Random Forest, Stemming, No Stop Words"))
print "Accuracy of Random Forest, nltk Stemming w/stop: %0.5f" % (test_forest(data, 0, 1, "Learning Curve Random Forest, NLTK Stemming, Stop Words"))
print "Accuracy of Random Forest, nltk Stemming w/o stop: %0.5f" % (test_forest(data, 2, 1, "Learning Curve Random Forest, NLTK Stemming, No Stop Words"))

plt.show()
