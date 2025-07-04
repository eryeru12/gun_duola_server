from gevent import monkey
monkey.patch_all()

from flask import Flask
app = Flask(__name__)

@app.route('/')
def hello():
    return "Hello, World! This is a Gunicorn server with gevent."

if __name__ == '__main__':
    app.run()
