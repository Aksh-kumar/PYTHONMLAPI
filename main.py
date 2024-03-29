#!/usr/bin/env python
# coding: utf-8

import os
import json
import io
import ast
from flask import request,Flask, jsonify
from flask_cors import CORS, cross_origin
from ML_algorithms.cluster.Expectation_Maximization import em_business as emb
from ML_algorithms.supporting_module import pickle_module as spr
import application_constant as CONSTANT
from logger import LOGGER

# to run flask
# in powershell $env:FLASK_APP = "main"
# set FLASK_ENV=development for development	
# set FLASK_APP=main.py
# python -m flask run

app = Flask(__name__)
CORS(app)

# dic to save Model
dic_model = {}


def get_model(k):
	"""
		To get EM model based on cluster number

		args :
			k (int) : number of cluster

		Returns :
			EMBusiness object
	"""
	
	if k in dic_model :
		return dic_model[k] # if already present in memory
	
	if CONSTANT.USED_SAVED_MODEL_FOR_4_CLUSTER and k == 4 : # return saved model for 4 cluster
		pklobj = emb.get_em_object(k, CONSTANT.TRAINING_PATH_DIR_EM, seed=CONSTANT.SEED, saved_model_name=CONSTANT.SAVED_MODEL_NAME)
	else :
		pklobj = emb.get_em_object(k, CONSTANT.TRAINING_PATH_DIR_EM, seed=CONSTANT.SEED)
	
	if pklobj is None : # If No model found
		raise Exception('no pickle object found')
	
	emobj = pklobj.pickled_object
	dic_model[k] = emobj
	return emobj


@app.route('/')
@cross_origin()
def base() :
	return str(__name__)


@app.route('/em/predict/', methods=['POST'])
@cross_origin()
def em_predict() :
	""" used to assigned label of cluster the Image
		coming from client side and return the result
	 	object contain image feature, responsibility
	 	and assigned cluster

		Request Method : 
			POST
		
		Body :
			type = 'application/json'
			value (str) : base64 encoded value of image
			filename (str) : name of Image file
			filetype (str) : Image extension
			k (int) : number of clusters (explicitly type cast to int)

		Return :
			Json Predicted responsibilities for all clusters along with the file details
	"""
	try :
		req = request.get_data().decode('utf-8')
		img = json.loads(req)
		img_base64 = img['value']
		img_name = img['filename']
		filetype = img['filetype']
		k = int(img['k'])
		path = os.path.join(CONSTANT.TEMP_FILE_PATH, img_name)
		path = spr.decode_base64(img_base64, path)
		if path is not None :
			emobj = get_model(k)
			result = emobj.predict_data(img_name, filetype, path, img_base64)
			return result.to_json(orient='records')
	except Exception as e:
		LOGGER.LOG(e) # log exception
		return {}


@app.route('/em/getclustername/', methods=['GET'])
@cross_origin()
def getclustername() :
	"""
		get cluster number to name mapping
		
		Method :
			GET
		
		Arguments :
			k (int) : number of clusters
		
		Return :
			cluster number to name mapping json
	"""
	
	try :
		k = request.args.get('k', type = int)
		emobj = get_model(k)
		return emobj.cluster_name
	except Exception as e:
		LOGGER.LOG(e) # Log Exception
		return {}


@app.route('/em/setclustername/', methods=['POST'])
@cross_origin()
def setclustername():
	"""
		set cluster number to name mapping

		Method :
			POST
		
		Body :
			type = 'application/json'
			k (int) : cluster number (explicitly casted)
			mappedey (dictionary) : cluster number to name mapping
		
		Return :
			JSON {'res' : (bool)saved status}
	"""
	try:
		req = request.get_data().decode('utf-8')
		data = json.loads(req)
		k = data['k']
		dic = data['mappedKey']
		emobj = get_model(k)
		emobj.set_cluster_name(dic)
		return emobj.set_cluster_name(dic)
	except Exception as e:
		LOGGER.LOG(e) # Log Exception
		return {}


@app.route('/em/getclusterparameter/', methods=['GET'])
@cross_origin()
def getclusterparameter():
	""" get all cluster trained parameter like weights
		means covarience metrix responsibilities for 
		each cluster

		Method :
			GET
		
		Arguments :
			k (int) : number of cluster (explicitly casted to int)

		Return :
			JSON contain {weights: weight of cluster, means: mean points of cluster, covariances: covariance metrix}
	"""
	try:
		k = request.args.get('k', type = int)
		emobj = get_model(k)
		param = emobj.em_parameters
		param['weights'] = param['weights'].tolist()
		param['means'] = param['means'].tolist()
		param['covariances'] = param['covariances'].tolist()
		del param['responsibility']
		return json.loads(json.dumps(param))
	except Exception as e:
		LOGGER.LOG(e) # Log Exception
		return {}


@app.route('/em/getsupportedimagesextension/', methods=['GET'])
@cross_origin()
def getsupportedimagesextension() :
	"""
		Used to get all the image file extension
		supported for image processing in EM algorithms

		Method :
			GET
		
		Arguments :
			k (int) : number of cluster

		Return :
			Array of string having supported extensions
	"""
	
	try :
		k = request.args.get('k', type = int)
		emobj = get_model(k)
		return jsonify(emobj.IMAGE_EXT_SUPPORTED)
	except Exception as e:
		LOGGER.LOG(e) # Log Exception
		return {}


@app.route('/em/getfirstndataresponsibility/', methods=['GET'])
@cross_origin()
def getfirstndataresponsibility():
	"""
		return all the image with parameters like
		responsibility image parameters assigned clusters

		Method :
			GET
		
		Arguments :
			k (int) : number of clusters
			n (int) : number of responsibilities
		
		Return :
			JSON With cluster name to top n requested training data responsibility with Details
	"""
	
	try:
		k = request.args.get('k', type = int)
		n = request.args.get('n', type = int)
		emobj = get_model(k)
		temp = emobj.get_first_n_data_responsibility(n, to_json=True)
		for i in temp.keys() :
			temp[i] = ast.literal_eval(temp[i]) # convert string list to list
		return  json.loads(json.dumps(temp))
	except Exception as e:
		LOGGER.LOG(e) # Log Exception
		return {}


@app.route('/em/getfirstnheterogeneity/', methods=['GET'])
@cross_origin()
def getfirstnheterogeneity():
	"""
		return value of k and its assiciate
		heterogeneity to obtain optimum value of k

		Method :
			GET
		
		Arguments :
			k (int) : number of clusters
			n (int) : number of First n heterogeneity

		Return :
			JSON of number of cluster to heterogeneity mapped pair
	"""
	
	try:
		k = request.args.get('k', type = int)
		n = request.args.get('n', type = int)
		emobj = get_model(k)
		return emobj.get_first_n_heterogeneity(n, seed=CONSTANT.SEED)
	except Exception as e:
		LOGGER.LOG(e)
		return {}


@app.route('/em/changek/', methods=['POST'])
@cross_origin()
def changek():
	"""
		change value of k and retrain the model

		Method :
			POST
		
		Arguments :
			k (int) : number of clusters

		Return :
			JSON {'res': (bool)status if successfully create model for changed k or not} 
	"""
	
	try:
		k = request.args.get('k')
		if k in dic_model :
			del dic_model[k]
		emObj = get_model(k)
		if emObj is None :
			return jsonify({'res':False})
		else :
			return jsonify({'res':True}) 
	except Exception as e:
		LOGGER.LOG(e) # Log Exception
		return {}



# main method start execution either in production or development mode

if __name__ == '__main__':
	app.run(host='127.0.0.1', port=5000, debug=True, threaded=True)
else:
    app.config.update(
        #SERVER_NAME='snip.snip.com:80',
        APPLICATION_ROOT='/',
    )