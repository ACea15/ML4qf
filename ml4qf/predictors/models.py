import importlib
import abc

sklearn_engines = ["neighbors", "neural_network", "semi_supervised", "svm",
                   "tree", "ensemble", "naive_bayes"]

class Model_factory:

    
    def __init__(self, library, engine_type, engine, engine_settings):
        """Build sklearn models on demand.

        Parameters
        ----------
        library : str
            Library used to build the model, e.g. keras, sklearn, etc.
        engine_type : str
            Type of technique, e.g. tree, svm etc
        engine : str
            The regressor or classifier name, e.g. SVC
        engine_settings : dict
            settings when building the model


        """
        self.library = library
        self.engine_type = engine_type
        self.engine = engine
        self.engine_settings = engine_settings
        self.model = None
        self.build()
        
    def build(self):

        if self.library == 'keras':
            from ml4qf.predictors.model_keras import Model_keras
            self.model = Model_keras(**self.engine_settings)
            self.engine_type = 'tailored keras'
            self.engine = 'deep_learning'
        elif self.library == 'scikit':
            module = importlib.import_module(f'sklearn.{self.engine_type}')
            engine_class = getattr(module, self.engine)
            self.model = engine_class(**self.engine_settings)
        else:
            raise NameError("library %s not implemented" % self.library)
        
    def __call__(self):

        return self.model

class Model(abc.ABC):
    """ """

    @abc.abstractmethod
    def build_model(self):
        pass

    @abc.abstractmethod
    def fit(self):
        pass

    # @abc.abstractmethod
    # def predict(self):
    #     pass

class Interface():

    pass

def factory_models():
    pass
