import main as mn
import traceback as tb
"""
	Basic test cases are written customly
"""

if __name__ == '__main__' :
	try:
		k = 4
		n = 5 # for get first n resp
		emobj = mn.get_model(k)
		if emobj is None :
			raise Exception('no pickle object found')
		print(emobj.em_parameters)
		print('cluster name')
		print(emobj.cluster_name)
		print('image supported extension')
		print(emobj.IMAGE_EXT_SUPPORTED)
		print('get forst n responsibility')
		print(emobj.get_first_n_data_responsibility(n, to_json=True))
		print('first n heteroginity')
		print(emobj.get_first_n_heterogeneity(n, seed=mn.CONSTANT.SEED))
		print('change k')
		emobj2 = mn.get_model(3)
		print(emobj2.get_em_params)
	except Exception as e:
		print(tb.format_tb(''.join(tb.format_tb(e.__traceback__))))
		mn.LOGGER.LOG(e)
    