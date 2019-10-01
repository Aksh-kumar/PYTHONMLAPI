from flask import request,Flask
from flask_cors import CORS, cross_origin
import os, json, io, ast
from ML_algorithms.cluster.EM import em_business as emb
from ML_algorithms.supporting_module import pickle_module as spr
""" used to initiate the cluster parameter like initial centroids in
k mean ++ algorithm it need to be constant trough out the model if
this seed s changed then recreate the model and pickled it otherwise
you will get different result"""
SEED = 0
# INIT Important class
TRAINING_PATH_DIR_EM = os.path.join(os.getcwd(), r'Data\EM\images')
CURRENT_DIR = os.getcwd()
TEMP_FILE_PATH = os.path.join(CURRENT_DIR, r'Temp')
app = Flask(__name__)
CORS(app)
# in powershell $env:FLASK_APP = "main"
# set FLASK_ENV=development for development	
# set FLASK_APP=main.py
# python -m flask run
# dic to save Model
dic_model = {}
def get_model(k) :
	global TRAINING_PATH_DIR_EM
	global SEED
	global dic_model
	if k in dic_model :
		return dic_model[k]
	pklobj = emb.get_em_object(k, TRAINING_PATH_DIR_EM, seed=SEED)
	if pklobj is None :
		raise Exception('no pickle object found')
	emobj = pklobj.pickled_object
	dic_model[k] = emobj
	return emobj
# End
@app.route('/')
@cross_origin()
def hello() :
	return str(__name__)
@app.route('/em/predict/', methods=['POST'])
@cross_origin()
def em_predict() :
	if request.method == 'POST':
		try :
			req = request.get_data().decode('utf-8')
			img = json.loads(req)
			#img = ft['Image']
			img_base64 = img['value']
			img_name = img['filename']
			filetype = img['filetype']
			k = int(img['k'])
			path = os.path.join(TEMP_FILE_PATH, img_name)
			path = spr.decode_base64(img_base64, path)
			if path is not None :
				emobj = get_model(k)
				result = emobj.predict_data(img_name, filetype, path, img_base64)
				return result.to_json(orient='records')[0]
		except :
			pass
	return {}
# End
@app.route('/em/getclustername/', methods=['GET'])
@cross_origin()
def getclustername() :
	try :
		k = request.args.get('k', type = int)
		emobj = get_model(k)
		print(emobj.cluster_name)
		return json.loads(json.dumps(emobj.cluster_name))
	except Exception as e :
		print(str(e))
		return {}
# End
@app.route('/em/setclustername/', methods=['POST'])
@cross_origin()
def setclustername() :
	try :
		req = request.get_data().decode('utf-8')
		data = json.loads(req)
		k = data['k']
		dic = data['mappedKey']
		emobj = get_model(k)
		emobj.cluster_name = dic
		emb.write_em_pickle(k, TRAINING_PATH_DIR_EM, SEED)
		del dic_model[k]
		emobj = get_model(k)
		return emobj.cluster_name
	except :
		return {}
# End
@app.route('/em/getclusterparameter/', methods=['GET'])
@cross_origin()
def getclusterparameter() :
	try :
		k = request.args.get('k', type = int)
		emobj = get_model(k)
		param = emobj.em_parameters
		param['weights'] = param['weights'].tolist()
		param['means'] = param['means'].tolist()
		param['covariances'] = param['covariances'].tolist()
		del param['responsibility']
		return json.loads(json.dumps(param))
	except :
		return {}
# End
@app.route('/em/getsupportedimagesextension/', methods=['GET'])
@cross_origin()
def getsupportedimagesextension() :
	k = request.args.get('k', type = int)
	emobj = get_model(k)
	return json.loads(json.dumps(emobj.IMAGE_EXT_SUPPORTED))
# End
@app.route('/em/getfirstndataresponsibility/', methods=['GET'])
@cross_origin()
def getfirstndataresponsibility() :
	try :
		k = request.args.get('k', type = int)
		n = request.args.get('n', type = int)
		emobj = get_model(k)
		temp = emobj.get_first_n_data_responsibility(n, to_json=True)
		for i in temp.keys() :
			temp[i] = ast.literal_eval(temp[i])
		return  json.loads(json.dumps(temp))
	except Exception as e:
		return {}
# End
@app.route('/em/getfirstnheterogeneity/', methods=['GET'])
@cross_origin()
def getfirstnheterogeneity() :
	try :
		k = request.args.get('k', type = int)
		n = request.args.get('n', type = int)
		emobj = get_model(k)
		return emobj.get_first_n_heterogeneity(n, seed=SEED)
	except :
		return {}
# End
@app.route('/em/changek/', methods=['POST'])
@cross_origin()
def changek() :
	try :
		k = request.args.get('k')
		obj = get_model(k)
		return {res:True}
	except :
		return {}
# End
if __name__ == '__main__':
	app.run(debug=True)
else:
    app.config.update(
        #SERVER_NAME='snip.snip.com:80',
        APPLICATION_ROOT='/',
    )