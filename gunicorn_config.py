bind = "0.0.0.0:8000"
workers = 1
worker_class = "gevent"
timeout = 300
keepalive = 5
preload_app = True
loglevel = "debug"  # 添加详细日志
spew = True  # 跟踪所有Python调用
