import sys
import os
import time
import pandas as pd
import numpy as np
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
from sklearn.pipeline import Pipeline


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
	return np.mean(predicted == Y_test)


def test_svm(stemmed=0, csv_name='temp.csv'):
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
			, max_iter=20
			, tol=1e-3
			, random_state=42))
	])

	text_clf_svm = text_clf_svm.fit(X_train, Y_train)
	predicted = text_clf_svm.predict(X_test)
	return np.mean(predicted == Y_test)


def save_reddit(posts, subreddits):
	postlist = []
	for subreddit in subreddits:
		print "Downloading submissions for ", subreddit
		postlist = postlist + sew(subreddit, posts)
		print "Concatenated postlist now contains ", len(postlist), " posts"
	reap(postlist)


# save_reddit(100, _subreddits)
print "Accuracy of Naive Bayes: %0.5f" % (test_nb())
print "Accuracy of SVM - stemming: %0.5f" % (test_svm())
print "Accuracy of SVM - no stemming: %0.5f" % (test_svm(1))
