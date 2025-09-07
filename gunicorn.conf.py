# gunicorn.conf.py
import os

workers = 1
worker_class = 'sync'
worker_connections = 1000
timeout = 30
keepalive = 2
errorlog = '-'
loglevel = 'debug'
accesslog = '-'