from sklearn.datasets import load_iris, load_boston
import xgboost as xgb
import xgbfir
import numpy as np

# loading database
boston = load_boston()
'''
Here 
'''
keys = list(boston.keys())
print(keys)
print(boston['feature_names'])
print(boston['filename'])
data_array = boston['data']
target = boston['target']
print(np.shape(data_array))
print(np.shape(target))
exit()

# doing all the XGBoost magic
xgb_rmodel = xgb.XGBRegressor().fit(boston['data'], boston['target'])

# saving to file with proper feature names
xgbfir.saveXgbFI(xgb_rmodel, feature_names=boston.feature_names, OutputXlsxFile='bostonFI.xlsx')


# loading database
iris = load_iris()

# doing all the XGBoost magic
xgb_cmodel = xgb.XGBClassifier().fit(iris['data'], iris['target'])

# saving to file with proper feature names
xgbfir.saveXgbFI(xgb_cmodel, feature_names=iris.feature_names, OutputXlsxFile='irisFI.xlsx')