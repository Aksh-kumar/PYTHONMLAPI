from flask import Flask
from flask_cors import CORS, cross_origin
app = Flask(__name__)
CORS(app)
# set FLASK_APP=FlaskHelloWorld.py
# python -m flask run
@app.route('/')
def hello() :
	return str(__name__)