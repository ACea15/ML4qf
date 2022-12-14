import utils
#from xgboost import (XGBClassifier, XGBRegressor, Booster)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (AdaBoostClassifier, AdaBoostRegressor, BaggingClassifier,
                              BaggingRegressor, GradientBoostingClassifier,
                              GradientBoostingRegressor, RandomForestClassifier,
                              RandomForestRegressor)
from sklearn.svm import (LinearSVC, LinearSVR, NuSVC, NuSVR, OneClassSVM, SVC, SVR)
from sklearn import tree

from sklearn.neighbors import KNeighborsClassifier

#models_dict = utils.get_predictors(locals())


