import sklearn.model_selection

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
