import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from datetime import datetime, timedelta
import xgboost as xgb
import operator
import matplotlib.pyplot as plt

train = pd.read_csv('train_withextra.csv')
test = pd.read_csv('test_withextra.csv')

#rmspe计算以及赋值给xgboost
def rmspe(y, yhat):
    return np.sqrt(np.mean((yhat/y-1) ** 2))

def rmspe_xg(yhat, y):
    y = np.expm1(y.get_label())
    yhat = np.expm1(yhat)
    return "rmspe", rmspe(y,yhat)

train = train[train["Open"] != 0]
train = train[train["Sales"] > 0]

print('training beging ................')

params = {'objective': 'reg:linear',
          'min_child_weight': 50,
          'booster' : 'gbtree',
          'eta': 0.05,
          'alpha': 2,
          'gamma': 2,
          'max_depth': 10,
          'subsample': 0.9,
          'colsample_bytree': 0.7,
          'silent': 1,
          'seed': 1301
          }

params['gpu_id'] = 0
params['tree_method'] = 'gpu_hist'

num_boost_round = 5000

print("Train a XGBoost model")

features = list(train.drop(['Sales','Customers'], axis=1))

#X_train, X_valid = train_test_split(train, test_size=0.02, random_state=1)

#调整训练集,验证集取最近的六周,去掉12月的
X_valid = train[(train['Year']==2015) & (train['WeekOfYear'] <=31) & (train['WeekOfYear'] >=26)]

X_train = train[(train['Year']==2015) & (train['WeekOfYear'] <26)] + train[(train['Year']!=2015)]

X_train = train[train['Month']!=12]
y_train = np.log1p(X_train.Sales)
y_valid = np.log1p(X_valid.Sales)
dtrain = xgb.DMatrix(X_train[features], y_train)
dvalid = xgb.DMatrix(X_valid[features], y_valid)

watchlist = [(dtrain, 'train'), (dvalid, 'eval')]

gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist, \
  early_stopping_rounds=100, 
  feval=rmspe_xg, 
  verbose_eval=True)


print("Validating")
yhat = gbm.predict(xgb.DMatrix(X_valid[features]))
error = rmspe(X_valid.Sales.values, np.expm1(yhat))
print('RMSPE: {:.6f}'.format(error))

print("Make predictions on the test set")
dtest = xgb.DMatrix(test[features])
test_probs = gbm.predict(dtest)

# 进行预测
result = pd.DataFrame({"Id": test["Id"], 'Sales': np.expm1(test_probs)*0.98})
result.to_csv('max_depth_10withextra.csv', index=False)

#gbm.save_model('5000_k_25max_depth_11withextra.model') 



#绘制特征图
def create_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    for i, feat in enumerate(features):
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
    outfile.close()

create_feature_map(features)
importance = gbm.get_fscore(fmap='xgb.fmap')
importance = sorted(importance.items(), key=operator.itemgetter(1))

df = pd.DataFrame(importance, columns=['feature', 'fscore'])
df['fscore'] = df['fscore'] / df['fscore'].sum() 

featp = df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(6, 10))
plt.title('XGBoost Feature Importance')
plt.xlabel('relative importance')
fig_featp = featp.get_figure() 

