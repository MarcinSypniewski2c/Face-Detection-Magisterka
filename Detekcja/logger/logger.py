import logging
import os
import sys
from logging.handlers import RotatingFileHandler

DEBUG_LOG_LOCATION='/tmp/face-reco-logs/'
ERROR_LOG_LOCATION='data/face-reco-logs/'

os.makedirs(DEBUG_LOG_LOCATION, exist_ok=True)
os.makedirs(ERROR_LOG_LOCATION, exist_ok=True)

logger =  logging.getLogger('main')
logger.setLevel(logging.DEBUG)

# persistent log for erros only
errors = RotatingFileHandler(ERROR_LOG_LOCATION + 'errors.log', maxBytes=2*1024*1024, backupCount=3, encoding='utf8')
errors.setLevel(logging.ERROR)

# temporary log in tmp, removed on device reboot
debug = RotatingFileHandler(DEBUG_LOG_LOCATION + 'debug.log', maxBytes=5*1024*1024, backupCount=5, encoding='utf8')
debug.setLevel(logging.DEBUG)

# console output
console = logging.StreamHandler()
console.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(module)s - %(message)s')
errors.setFormatter(formatter)
debug.setFormatter(formatter)
console.setFormatter(formatter)

logger.addHandler(errors)
logger.addHandler(debug)
logger.addHandler(console)