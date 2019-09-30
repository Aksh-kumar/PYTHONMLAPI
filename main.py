from flask import request,Flask
from flask_cors import CORS, cross_origin
import os, json, io
from ML_algorithms.cluster.EM import em_business as emb
from ML_algorithms.supporting_module import pickle_module as spr
# INIT Important class
TRAINING_PATH_DIR_EM = os.path.join(os.getcwd(), r'\Data\EM\images')
CURRENT_DIR = os.getcwd()
TEMP_FILE_PATH = os.path.join(CURRENT_DIR, r'Temp')
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
	res = False
	if request.method == 'POST':
		try :
			req = request.get_data().decode('utf-8')
			img = json.loads(req)
			#img = ft['Image']
			img_base64 = img['value']
			img_name = img['filename']
			path = os.path.join(TEMP_FILE_PATH, img_name)
			res = spr.decode_base64(img_base64, path)
		except :
			pass
	return {'sucess': res}
if __name__ == '__main__':
	app.run(debug=True)
else:
    app.config.update(
        #SERVER_NAME='snip.snip.com:80',
        APPLICATION_ROOT='/',
    )