import sys
import os
sys.path.append(os.path.abspath("C:\\Users\\tchisholm\\.PyCharm2018.3\\config\\scratches\\"))
from reddit_id import *

reddit = get_reddit()
submission_num = 0
separator = '----------------------------------------------------------------------'


for submission in reddit.subreddit('legaladvice').stream.submissions():
	submission_num = submission_num + 1
	if submission_num > 10:
		exit(0)
	submission.comments.replace_more(limit=None)
	for comment in submission.comments.list():
		# print(comment.body)
		# print(comment.name)
		# print(comment.author)
		if str(comment.author) == 'LocationBot':
			continue
		print(separator + '\n' + 'AUTHOR: ' + str(comment.author) + '\n' + comment.body + '\n' + separator + '\n')

