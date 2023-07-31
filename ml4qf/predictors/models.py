import importlib
import abc
import ml4qf.predictors.model_keras as model_keras

sklearn_engines = ["neighbors", "neural_network", "semi_supervised", "svm",
                   "tree", "ensemble", "naive_bayes"]

def model_factory(library, engine_type, engine, engine_settings) -> obj:
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

        if library == 'keras':
            Model_keras = getattr(model_keras, engine_type)
            model = Model_keras(**engine_settings)
            engine = 'keras'
        elif library == 'scikit':
            module = importlib.import_module(f'sklearn.{engine_type}')
            engine_class = getattr(module, engine)
            model = engine_class(**engine_settings)
        else:
            raise NameError("library %s not implemented" % library)

        return model

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

