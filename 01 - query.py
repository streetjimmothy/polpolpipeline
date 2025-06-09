#import modules used for performance profiling
import time
import humanize
import psutil
import datetime 
import argparse

#import database module
from pymongo import MongoClient, errors

parser = argparse.ArgumentParser(description="Runs community detection. Optionally saves central tweets to a file.")
parser.add_argument("-l", "--language", required=False, help="ISO language code (e.g. 'en' for English, 'es' for Spanish, etc.). Will be requested if not provided.")
parser.add_argument("-d", "--date-range",required=False, help="Date range from the valid options. Will be requested if not provided.")

args = parser.parse_args()

if args.language:
	language = args.language
else:
	language = input("select language: \n \t1. English\n \t2. Spanish\n \t3. French\n \t4. Italian\n \t5. Portuguese\n")
	if language == '1':
		language = 'en'
	elif language == '2':
		language = 'es'
	elif language == '3':
		language = 'fr'
	elif language == '4':
		language = 'it'
	elif language == '5':
		language = 'pt'

if args.date_range:
	date_range = args.date_range
else:
	date_range = input("select date range: \n \t1. March 2020 - June 2020\n \t2. June 2020 - Nov 2020\n \t3. Nov 2020 - Feb 2021\n \t4. 1+2\n \t5. 1+3\n \t6. 2+3\n \t7. All\n")
if date_range == '1':
	start_date = datetime.datetime(2020, 3, 11)
	end_date = datetime.datetime(2020, 6, 11)
elif date_range == '2':
	start_date = datetime.datetime(2020, 6, 11)
	end_date = datetime.datetime(2020, 11, 11)
elif date_range == '3':
	start_date = datetime.datetime(2020, 11, 11)
	end_date = datetime.datetime(2021, 2, 11)
elif date_range == '4':
	start_date = datetime.datetime(2020, 3, 11)
	end_date = datetime.datetime(2020, 11, 11)
elif date_range == '5':
	start_date = datetime.datetime(2020, 3, 11)
	end_date = datetime.datetime(2021, 2, 11)
elif date_range == '6':
	start_date = datetime.datetime(2020, 6, 11)
	end_date = datetime.datetime(2021, 2, 11)
elif date_range == '7':
	start_date = datetime.datetime(2020, 3, 11)
	end_date = datetime.datetime(2021, 2, 11)

#VARIABLES NEEDED TO CONNECT TO AND QUERY DATABASE

CONNECTION_STRING = "mongodb://JamIs:morticiaetpollito@118.138.244.29:27017/"

class Tweet:
	def __init__(self, tweet):
		self.id = tweet['id']
		self.user = tweet['user']
		self.connected_user = tweet['connected_user']
		self.connection_type = tweet['connection_type']
		self.text = None
		if('text' in tweet):
			self.text = tweet['text']
		if('fulltext' in tweet):
			self.text = tweet['fulltext']

query = {"$and":
			[
				{"datetime": {"$gte": start_date}},
				{"datetime": {"$lte": end_date}},
				{"lang": language},
				{"connection_type": {"$ne": None}}
			]
		}


start_time = time.time()

#CONNECT TO DATABASE
print("connecting...")
client = MongoClient(CONNECTION_STRING)
tw_coll = client.get_database('Tw_Covid_DB').get_collection('tweets')
tu_coll = client.get_database('Tw_Covid_DB').get_collection('users')
print("connected")

#RUN QUERY
query_results = tw_coll.find(query)
db_tweets = []
tweet_id_index = {}
i = 0 #could do a range thing below, but cbf
with open(f"./query_results/{language}_{start_date.year}-{start_date.month:02d}_{end_date.year}-{end_date.month:02d}.txt", "w", encoding="utf-8") as file:
	for t in query_results:
		tweet = Tweet(t)
		if tweet.text != None:
			db_tweets.append(tweet)
			tweet_id_index[tweet.id] = i
			i+=1
			if i%100000 ==0:
				print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ": " + str(i))
			file.write(str(t))
			file.write('\n')
print(len(db_tweets))

print("Time taken: {}".format(humanize.precisedelta(time.time() - start_time, suppress=['days', 'milliseconds', 'microseconds'])))
print("Memory used: {}".format(humanize.naturalsize(psutil.Process().memory_info().rss)))