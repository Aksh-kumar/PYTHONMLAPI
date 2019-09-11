from flask import request,Flask
from flask_cors import CORS, cross_origin
from cluster.EM import em_business
import os
app = Flask(__name__)
CORS(app)
# in powershell $env:FLASK_APP = "main"
# set FLASK_ENV=development for development	
# set FLASK_APP=main.py
# python -m flask run
@app.route('/')
def hello() :
	return str(__name__)
@app.route('/em', methods=['POST', 'GET'])
def em_predict() :
	if request.method == 'POST':
		f = dict(request.form)
		print(f)
		# f.save(os.path.join('./Temp/'+ f.filename))
		return True
	else :
		return False
if __name__ == '__main__':
	app.run(debug=True)
else:
    app.config.update(
        #SERVER_NAME='snip.snip.com:80',
        APPLICATION_ROOT='/',
    )