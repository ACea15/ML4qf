
class Main:

    def __init__(self):

        self.inputs = None

class Labels:

    def set_bintarget(df, label='price', alpha=0., check_iftarget=True,
                      remove_nan=True, print_info=True):
        """ Sets binary target"""

        target_set = False
        if 'Target' not in df.columns:
            df['Target'] = np.where(df[label].shift(-1) > (1 - alpha) * df[label], 1, 0)
            target_set = True
        elif not check_iftarget:
            df['Target'] = np.where(df[label].shift(-1) > (1 - alpha) * df[label], 1, 0)
            target_set = True
        if remove_nan and target_set:
            df = df[:-1]
        if print_info:
            df['Target'].count
        return df


    
import sklearn.model_selection
class HyperTuning:

    def __init__(self, predictor, searcher_name, hyper_grid, searcher_settings, cv_name=None, cv_settings=None):

        self.predictor = predictor
        self.searcher_name = searcher_name
        self.hyper_grid = hyper_grid
        self.searcher_settings = searcher_settings
        self.searcher = None
        if cv_name is None or isinstance(cv_name, int):
            self.cv = cv_name
        else:
            self.cv = CrossValidation(self.cv_name, self.cv_settings)        
        self.set_searcher()
        
    def set_searcher(self):

        self.searcher_type = getattr(sklearn.model_selection, self.searcher_name)
        self.searcher = self.searcher_type(self.predictor,
                                           self.hyper_grid,
                                           cv=self.cv,
                                           **self.searcher_settings)

class CrossValidation:


    def __init__(self,  cv_name, cv_settings):


        self.cv_name = cv_name
        self.cv_settings = cv_settings
        self.transformer = None
        self.set_crossvalidation()
        
    def set_crossvalidation(self):

        self.cv_type = getattr(sklearn.model_selection, self.cv_name)
        self.cv = self.cv_type(**self.cv_settings)
    
import sklearn.preprocessing
class Transformer:

    def __init__(self,  transformer_name, transformer_settings):

        
        self.transformer_name = transformer_name
        self.transformer_settings = transformer_settings
        self.transformer = None
        self.set_transformer()
        
    def set_transformer(self):

        self.transformer_type = getattr(sklearn.preprocessing, self.transformer_name)
        self.transformer = self.transformer_type(**self.transformer_settings)
