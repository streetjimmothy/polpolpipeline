#Connect to database

from pymongo import MongoClient
client = MongoClient("mongodb://JamIs:morticiaetpollito@118.138.244.29:27017/")
db = client.Tw_Covid_DB
tweets = db.tweets

dates = {}

for tweet in tweets.find():
	#check date of tweet
	messy_date = tweet["created_at"] #"Thu Mar 12 02:01:57 +0000 2020"
	real_date = messy_date[:10] + messy_date[-4:]

	if real_date in dates.keys():
		#increment counter for each date
		dates[real_date] += 1
	else:
		dates[real_date] = 1

best_date = None
for date in dates.keys():
	if best_date is None:
		best_date = date
	if dates[best_date] < dates[date]:
		best_date = date

print(best_date)

#Check everyday tweet count

