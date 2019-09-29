""" Used to preserve EMBusiness object as pickle file and some meta data like training files
 path and value of k which can be later on used"""
class EMPickledClass :
    def __init__(self, path, obj, k) :
        self._path = path
        self._obj = obj
        self._k = k
    @property
    def path(self) :
        """ return training path of contained model object"""
        return self._path
    @property
    def pickled_object(self) :
        """ returned model object """
        return self._obj
    @property
    def k(self) :
        """ return number of clusters in model objects"""
        return self._k
# End