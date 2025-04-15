import time
import logging
from pythonjsonlogger import jsonlogger
from logging.handlers import RotatingFileHandler

logger = logging.getLogger(' logs ')
logging.basicConfig(level=logging.INFO, filemode='a')
json_handler = RotatingFileHandler(f'logs/logs.json', maxBytes=100000000, backupCount=10)
json_handler.setLevel(logging.INFO)
json_handler.setFormatter(jsonlogger.JsonFormatter('%(asctime)s %(levelname)s'))
logger.addHandler(json_handler)

def log(log_type, status_code, client_addr, request_id, done, delays, error):
    to_log = {'status_code': status_code, 'client_addr': client_addr, 'request_id': request_id, 'success': done, 'time': delays, 'error': error}
    if log_type == 'INFO':
        logger.info(to_log)
    elif log_type == 'ERROR':
        logger.error(to_log)
    return to_log