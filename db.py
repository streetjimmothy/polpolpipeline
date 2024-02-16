import sys
import logging
logFormatter = logging.Formatter("%(asctime)s::%(levelname)s:%(message)s")
rootLogger = logging.getLogger()
rootLogger.setLevel(logging.DEBUG)

fileHandler = logging.FileHandler("db.log")
fileHandler.setFormatter(logFormatter)
rootLogger.addHandler(fileHandler)

consoleHandler = logging.StreamHandler(sys.stdout)
consoleHandler.setFormatter(logFormatter)
rootLogger.addHandler(consoleHandler)

import time
import humanize
import psutil

from bson.json_util import loads
import os
from pymongo import MongoClient
import re

CONNECTION_STRING = "mongodb://JamIs:morticiaetpollito@118.138.244.29:27017/"

class Tweet:
	def __init__(self, tweet):
		self.id = tweet['id']
		self.user = tweet['user']
		self.connected_user = tweet['connected_user']
		self.connection_type = tweet['connection_type']
		self.text = tweet['text']


class DB:
	def __init__(self):
		self.connect()
		self.tweets = {}
		self.query_results = None

	def connect(self):
		logging.info("connecting...")
		client = MongoClient(CONNECTION_STRING)
		self.tw_coll = client.get_database('Tw_Covid_DB').get_collection('tweets')
		self.tu_coll = client.get_database('Tw_Covid_DB').get_collection('users')
		logging.info("connected")

	def run_query(self):
		if self.query_results is None:
			logging.info("running query...")
			query = None
			try:
				if args.query is not None:
					query = args.query
				if args.query_file is not None:
					with open(args.query_file, 'r') as f:
						query = f.read()
			except:
				if query is None:
					with open(os.path.join(os.path.dirname(os.path.realpath(__file__)),"query.json"), 'r') as f:
						query = f.read()
			query = re.sub(r'\s+', ' ', query)
			self.query_results = self.tw_coll.find(loads(query))
		else:
			logging.info("Loading query from cache.")
		return self.query_results

	def post_process(self):
		if(len(self.tweets) != 0):
			logging.info("Loading post processed tweets from cache.")
			return
		
		tweets = {}
		final_tweets ={}
		user_tweet_mapping = {}

		logging.info("processing tweets...")
		start_time = time.time()
		try:
			for t in self.query_results:
				#we do a mapping of tweets to users this way so we can filter users based on ...location? eventually
				#user_tweet_mapping is an interim mapping we use to get the users from the database. At query time we add any additional criteria
				#tweets is a interm list of Tweets objects so we can keep the tweet id, users id, and connected_user id bundled during the batched query
				tweet = Tweet(t)
				tweets[tweet.id] = tweet
				if tweet.user not in user_tweet_mapping:
					user_tweet_mapping[tweet.user] = []
				if tweet.connected_user not in user_tweet_mapping:
					user_tweet_mapping[tweet.connected_user] = []
				user_tweet_mapping[tweet.user].append(tweet.id) 
				user_tweet_mapping[tweet.connected_user].append(tweet.id)
				
				if len(user_tweet_mapping) > 4096:
					user_list = user_tweet_mapping.keys()
					users = self.tu_coll.find({"_id":{"$in":list(user_list)}})#list()}}) #get the users
					for user in users:
						# if the user is found by our constrained query, the Tweet object with the bundled tweet id, users id, and connected_user id
						# is added to the final_tweets list. The final_tweets list therefore only has tweets by users who meet our criteria
						for tweet_id in user_tweet_mapping[user['id']]:
							final_tweets[tweet_id] = tweets[tweet_id]
					user_tweet_mapping = {} #reset
		except Exception as e:
			logging.error(e)
		finally:
			end_time = time.time()
			self.tweets = final_tweets
			logging.info("Tweets loaded: {}".format(len(self.tweets)))
			logging.info("Time taken: {}".format(humanize.precisedelta(end_time - start_time, suppress=['days', 'milliseconds', 'microseconds'])))
			logging.info("Memory used: {}".format(humanize.naturalsize(psutil.Process().memory_info().rss)))
