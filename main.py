from flask import request,Flask
from flask_cors import CORS, cross_origin
from cluster.EM import em_business
import os, base64, json, io
app = Flask(__name__)
CORS(app)
# in powershell $env:FLASK_APP = "main"
# set FLASK_ENV=development for development	
# set FLASK_APP=main.py
# python -m flask run
@app.route('/')
@cross_origin()
def hello() :
	return str(__name__)
@app.route('/em/predict/', methods=['POST'])
@cross_origin()
def em_predict() :
	if request.method == 'POST':
		req = request.get_data().decode('utf-8')
		ft = json.loads(req)
		img = ft['Image']
		img_base64 = img['value']
		img_name = img['filename']
		img_str = base64.b64decode(img_base64)
		with open('./Temp/'+ img_name, 'wb') as f :
			f.write(img_str)
		return {'sucess': True}
	else :
		return False
if __name__ == '__main__':
	app.run(debug=True)
else:
    app.config.update(
        #SERVER_NAME='snip.snip.com:80',
        APPLICATION_ROOT='/',
    )