import sys
import logging
logFormatter = logging.Formatter("%(asctime)s::%(levelname)s:%(message)s")
rootLogger = logging.getLogger()
rootLogger.setLevel(logging.DEBUG)

fileHandler = logging.FileHandler("pipeline.log")
fileHandler.setFormatter(logFormatter)
rootLogger.addHandler(fileHandler)

consoleHandler = logging.StreamHandler(sys.stdout)
consoleHandler.setFormatter(logFormatter)
rootLogger.addHandler(consoleHandler)

from db import DB

DB = DB()
DB.run_query()
DB.post_process()

#iterate over all folders in the target directory
# need to make them packages in build.py
# path = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../polpolpipeline/"))
# 	sys.path.append(path)
# 	import db