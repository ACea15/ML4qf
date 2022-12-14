import numpy as np
import os
import pandas as pd

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)

from models import models_dict

class ModelPrediction:

    """
    Build predictor model
    """
    
    def __init__(self, predictor_name, **kwargs):

        self.settings = kwargs
        self.predictor_name = predictor_name
        self.predictor = models_dict[predictor_name](**self.settings)

    def run(self, X_train=None, y_train=None, X_test=None, y_test=None,
            print_info=True, **kwargs):

        # fit the predictor model
        if X_train is not None and y_train is not None:
            self.Xtrain = X_train
            self.ytrain = y_train
            self.predictor.fit(X_train, y_train)
            self.ytrain_pred = self.predictor.predict(X_train)
        if X_test is not None and y_test is not None:
            self.Xtest = X_test
            self.ytest = y_test
            self.ytest_pred = self.predictor.predict(X_test)

        # Classification Report
        if print_info:

            self.predictor_info()
            
    def set_signal(self, label, df_train=None, df_test=None):

        return_df = []
        if df_train is not None:
            df_train[''.join(['signal_', label])] = self.ytrain_pred
            return_df.append(df_train)
        if df_test is not None:
            df_test[''.join(['signal_', label])] = self.ytest_pred
            return_df.append(df_test)
        if len(return_df) == 1:
            return return_df[0]
        elif len(return_df) == 2:
            return return_df
        else:
            raise ValueError("No df provided")
        
    def predictor_info(self):

        if self.Xtrain is not None and self.ytrain is not None:
            print('#####################################################')
            print('####### Classification_report (Training data) #######')
            print('#####################################################')
            print(classification_report(self.ytrain, self.ytrain_pred))
            print('#####################################################')
        if self.Xtrain is not None and self.ytrain is not None:
            print('####### Classification_report (Testing data)  #######')
            print('#####################################################')
            print(classification_report(self.ytest, self.ytest_pred))
            print('#####################################################')

class ModelHyperTuning:


    def __init__(self, predictor_name, searcher_name, hyper_grid,
                predictor_settings, searcher_settings, cv_settings):

        self.model = ModelPrediction(predictor_name, **predictor_settings)
        self.tscv = split_timedata(X=None, y=None, output_splitdata=False, **cv_settings)
        self.searcher_type = getattr(model_selection, searcher_name)

        self.searcher = self.searcher_type(self.model.predictor,
                                           hyper_grid,
                                           cv=self.tscv,
                                           **searcher_settings)
        self.predictor_name = predictor_name
        self.searcher_name = searcher_name
        self.hyper_grid = hyper_grid
        self.searcher_settings = searcher_settings
        self.predictor_settings = predictor_settings

    @property
    def model_params(self):

        # Get params list
        return self.model.predictor.get_params()

    def run(self, X_train, y_train, print_info=True, **kwargs):

        self.searcher.fit(X_train, y_train, **kwargs)

        if print_info:
            # best parameters
            print(self.searcher.best_params_)
            # best score
            print(self.searcher.best_score_)

    def new(self, X_train=None, y_train=None, X_test=None, y_test=None,
            print_info=True, **kwargs):
        # Refit the XGB Classifier with the best params
        self.newmodel = ModelPrediction(self.predictor_name, **self.searcher.best_params_)
        self.newmodel.run(X_train, y_train, X_test, y_test, print_info, **kwargs)
