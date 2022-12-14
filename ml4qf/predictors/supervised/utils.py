def get_predictors(input_locals):

    predictors_dict = dict()
    for k, v in input_locals.items():

        if ('Classifier'  in k or
            'Regressor'  in k or
            'SVC'         in k or
            'SVR'         in k or
            'SVM'         in k):
            
            predictors_dict[k] = v
            
    return predictors_dict
