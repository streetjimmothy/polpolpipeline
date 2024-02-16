import sys
import logging
logFormatter = logging.Formatter("%(asctime)s::%(levelname)s:%(message)s")
rootLogger = logging.getLogger()
rootLogger.setLevel(logging.DEBUG)

fileHandler = logging.FileHandler("build.log")
fileHandler.setFormatter(logFormatter)
rootLogger.addHandler(fileHandler)

consoleHandler = logging.StreamHandler(sys.stdout)
consoleHandler.setFormatter(logFormatter)
rootLogger.addHandler(consoleHandler)

import subprocess

#instead of doing this, go through each folder and get the requirements file
subprocess.check_call([sys.executable, "-m", "pip", "install", "humanize"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "pymongo"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "networkx"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "igraph"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "leidenalg"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "sklearn"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "psutil"])

#might be necessary to install pycairo
# sudo apt-get update -y
# sudo apt-get install -y pkg-config
# sudo apt-get install libcairo2-dev libjpeg-dev libgif-dev
# sudo pip install pycairo
subprocess.check_call([sys.executable, "-m", "pip", "install", "pycairo"])