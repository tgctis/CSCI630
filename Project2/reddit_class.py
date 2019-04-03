import sys
import os
import time
import pandas as pd
import hashlib
sys.path.append(os.path.abspath("C:\\tmp"))
from reddit_id import *

reddit = get_reddit()


def load_df(_subreddit):
	return pd.read_csv(_subreddit + ".csv")


def word_sack(_df):
	return _df.apply(lambda x: x.str.split(expand=True).stack()).stack().value_counts()

def cleave(_subreddit, MAX_POSTS):
	timestamp = time.time()
	postlist = []

	submission_num = 0
	for submission in reddit.subreddit(_subreddit).stream.submissions():
		submission_num = submission_num + 1
		if submission_num > MAX_POSTS:
			break
		print "Submission #", submission_num
		submission.comments.replace_more(limit=None)
		for comment in submission.comments.list():
			hash_object = hashlib.md5(comment.permalink)
			id = hash_object.hexdigest()
			post = [id, comment.body, _subreddit]
			postlist.append(post)

	print "Posts: ", len(postlist)
	df = pd.DataFrame(postlist, columns=['ID', 'BODY', 'SUBREDDIT'])
	df.to_csv(_subreddit + '.csv', index=False, mode='w', encoding="utf-8")


def workit(posts, subreddits):
	for subreddit in subreddits:
		# cleave(subreddit, posts)
		df = load_df(subreddit)
		bag_of_words = word_sack(df)
		print bag_of_words


workit(100, ['legaladvice', 'politics'])
