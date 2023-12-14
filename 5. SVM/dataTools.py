import numpy as np
import pandas as pd
import scipy as sp

import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LinearRegression

from itertools import combinations

def coef(model, x):
    tmp = []
    for coef in model.coef_:
        tmp.append('{:0.02f}'.format(coef))
  
    if isinstance(x, np.ndarray):
        x = pd.DataFrame(x)
    
    coef_ = pd.DataFrame( tmp, columns=['coef'] )
    col = pd.DataFrame( x.columns, columns=['columns'] )
    return pd.concat( [coef_, col], axis=1)

def rSquare( x, y, yhat ):
    if x.ndim == 1: p, n = 1, x.shape[0]
    else: p, n = x.shape[1], x.shape[0]
    r2 = 1 - np.sum( (y - yhat) ** 2) / np.sum( (y - np.mean(y)) ** 2 ) 
    adj_r2 = 1 - (1 - r2) * ( n - 1) / ( n - p - 1 )
    return {'r2': r2, 'adjr2': adj_r2}


def visualCorr(x, width=15, height=7):
    plt.figure(figsize=(width,height))
    mask = np.zeros_like(x.corr(), dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(x.corr(), annot=True, fmt='.2f', mask=mask, cmap='seismic')
    plt.show()


def vif(x):
    vifFrame = pd.DataFrame()
    vifFrame['vif factor'] = [variance_inflation_factor(x.values, i) for i in range(x.shape[1]) ]
    vifFrame['features'] = x.columns
    return vifFrame


def forward(model, x, y, selected_columns):
    forward_columns = [ col for col in x.columns if col not in selected_columns ]
    result = []
    for column in forward_columns:
        columns = selected_columns + [column]
        tmp = model.fit(x[columns], y)
        yhat = tmp.predict(x[columns])
        score = rSquare(x[columns], y, yhat)
        result.append( {'model': tmp, 'score': score['adjr2'], 'columns': columns} )
  
    models = pd.DataFrame(result)
    best_model = models.loc[ models.score.argmax() ]
    return best_model

def backward(model, x, y, selected_columns):
    result = []
    for combi in combinations(selected_columns, len(selected_columns)-1):
        columns = list(combi)
        tmp = model.fit(x[columns], y)
        yhat = tmp.predict(x[columns])
        score = rSquare(x[columns], y, yhat)
        result.append( {'model': tmp, 'score': score['adjr2'], 'columns': columns} )
  
    models = pd.DataFrame(result)
    best_model = models.loc[ models.score.argmax() ]
    return best_model

def forward_selection(x, y):
    selected_columns = []
  
    for i in range(0, x.shape[1]):
        model = LinearRegression()
        ret = forward(model, x, y, selected_columns)
  
        if not i:
            before_model = ret
        else: 
            if ret.score > before_model.score: before_model = ret
            else: break
        selected_columns = ret.columns
    return before_model

def backward_elimination(x, y):
    selected_columns = x.columns
  
    for i in range(0, x.shape[1]):
        model = LinearRegression()
        ret = backward(model, x, y, selected_columns)
  
        if not i:
            before_model = ret
        else: 
            if ret.score > before_model.score: before_model = ret
            else: break
        selected_columns = ret.columns
    return before_model

def stepwise_selection(x, y):
    selected_columns = []

    for i in range(x.shape[1]):
        model = LinearRegression()
        forward_model = forward(model, x, y, selected_columns)
        selected_columns = forward_model.columns

        if i < 1: before_model = forward_model; continue

        backward_model = backward(model, x, y, selected_columns)
        large_model = forward_model
        if forward_model.score < backward_model.score:
            selected_columns = backward_model.columns
            large_model = backward_model
    
        if large_model.score > before_model.score: before_model = large_model
        else: break
    return before_model

def dimensionchange(model, x1, x2, h=0.01, **params):
    xx1, xx2 = np.meshgrid(
        np.arange(x1.min() - 0.1, x1.max() + 0.1, h), 
        np.arange(x2.min() - 0.1 , x2.max() + 0.1 , h)
    )
    y = model.predict(np.c_[xx1.flatten(), xx2.flatten()])
    y = y.reshape(xx1.shape)
    result = plt.contourf(xx1, xx2, y, **params)
    return result
