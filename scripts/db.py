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

	def run_query(self, query = None, cache = False):
		if not cache or self.query_results is None:
			logging.info("running query...")
			# a query or queryfile on the commend line overrides one hardcoded
			try:
				if args.query is not None:
					query = args.query
			except:
				pass
			try:
				#if query is passed in or taken from the command line, attempt to openm it as a file
				if query is not None:
					with open(query, 'r') as f:
						#if it is a file, read the query from the file
						query = f.read()
			except:
				pass
			#either the query is passed in, taken from the command line, or read from file
			#if none of the above, read the default query
			if query is None:
				with open(os.path.join(os.path.dirname(os.path.realpath(__file__)),"query.json"), 'r') as f:
					query = f.read()
			if isinstance(query, str):
				query = re.sub(r'\s+', ' ', query) #remove extra spaces?
				query = loads(query)
			self.query_results = self.tw_coll.find(query, no_cursor_timeout=True)
		else:
			#This shouldn't actually be done, because what's in the cache is only the cursor, not the data...
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
