import numpy as np
import matplotlib.pyplot as plt

np.random.seed(410)
xx = np.linspace(-1, 1, 500)
x = np.linspace(-1, 1, 50)
y_train = x ** 2 + 0.5 * x + 1 + 0.5 * (2 * np.random.rand(len(x)) - 1)
y_test = x ** 2 + 0.5 * x + 1 + 0.5 * (2 * np.random.rand(len(x)) - 1)

plt.plot(x, y_train, 'o', label='train')
plt.plot(x, y_test, 'xr', label='test')
plt.grid()
plt.legend()
plt.show()
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
treeModel = DecisionTreeRegressor(max_depth=3).fit(x.reshape(-1,1),y_train)

plt.plot(x, y_train, 'o', label='train')
plt.plot(x, y_test, 'xr', label='test')
plt.plot(xx, treeModel.predict(xx.reshape(-1,1)))
plt.grid()
plt.legend()
plt.show()

treeModel.score(x.reshape(-1,1), y_train)
treeModel.feature_importances_
