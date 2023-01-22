import numpy as np
import sklearn.model_selection
import ml4qf.utils

class HyperTuning:

    def __init__(self, predictor, searcher_name, searcher_settings, hyper_grid,
                 cv_name=None, cv_settings=None):

        self.predictor = predictor
        self.searcher_name = searcher_name
        self.hyper_grid = hyper_grid
        self.searcher_settings = searcher_settings
        self.cv_name = cv_name
        self.cv_settings = cv_settings
        self.searcher = None
        self._searcher_type = None
        if isinstance(cv_name, int) or cv_name is None:
            self.cv = cv_name
        else:
            self._cv = CrossValidation(self.cv_name, self.cv_settings)
            self.cv = self._cv()
        self.set_searcher()
        
    def set_searcher(self):

        try:
            self._searcher_type = globals()[self.searcher_name]
            self.searcher = self._searcher_type(self.predictor,
                                                self.hyper_grid,
                                                cv=self.cv,
                                                **self.searcher_settings)
        except KeyError:
            self._searcher_type = getattr(sklearn.model_selection, self.searcher_name)
            self.searcher = self._searcher_type(self.predictor,
                                                self.hyper_grid,
                                                cv=self.cv,
                                                **self.searcher_settings)

    def __call__(self):

        return self.searcher

class CrossValidation:

    def __init__(self,  cv_name, cv_settings):

        self.cv_name = cv_name
        self.cv_settings = cv_settings
        self.cv = None
        self.set_crossvalidation()
        
    def set_crossvalidation(self):

        self._cv_type = getattr(sklearn.model_selection, self.cv_name)
        self.cv = self._cv_type(**self.cv_settings)
        
    def __call__(self):

        return self.cv
    
class Full_Grid:

    def __init__(self, predictor, hyper_grid, cv, fit_sett=None, **kwargs):
        
        self.predictor = predictor,
        self.hyper_grid = hyper_grid,
        self.cv = cv,
        self.settings = kwargs
        self.hyper_space = list(ml4qf.utils.product_dict(**hyper_grid))
        if fit_sett is not None:
            self.fit_settings = fit_sett
        else:
            self.fit_settings = dict()
            
    def scoring(self, X, y, calls=None):
        
        score = []
        for hi in self.hyper_space:
            #calls: tensorflow.keras.backend.clear_session()
            self.predictor.set_params(**hi)
            score_hi = []
            for cvi in self.cv.split(X):
                index_train, index_test = cvi
                Xtrain_i = X[index_train]
                ytrain_i = y[index_train]
                Xtest_i = X[index_test]
                self.predictor.fit(Xtrain_i, ytrain_i, **self.fit_settings)
                ypred = self.predictor.predict(Xtest_i)
                score_i = self.predictor.score(y[index_test], ypred)
                score_hi.append(score_i)
    
            score.append(np.average(score_hi))
    
        return score
