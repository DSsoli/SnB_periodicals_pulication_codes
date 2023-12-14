import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn.linear_model import LinearRegression

x = np.array([30, 31, 32, 33, 34])
y = np.array([19, 22, 32, 47, 52])

#Drawing line plots using matplotlib
plt.plot(x, y)
plt.grid()
plt.show()

#Drawing line plots using seaborn
sns.relplot(x=x, y=y, kind='line')
plt.grid()
plt.show()

#Drawing scatter plots using matplotlib (1)
plt.plot(x, y, 'o')
plt.grid()
plt.show()

#Drawing scatter plots using matplotlib (2)
plt.scatter(x, y)
plt.grid()
plt.show()

#Drawing scatter plots using seaborn
sns.relplot(x=x, y=y)
plt.grid()
plt.show()

# x와 y 사이의 관계를 찾으려는 것
# 즉, 두 집합 x, y 사이의 함수를 찾는 것. 집합론이 있기에 머신러닝/딥러닝이 가능
# Linear regression은 선형적인 관계가 있음을 가정
# 선형적인 관계가 있음을 가정할 경우 f(x) = ax + b
# 여기서 b가 y절편, a가 기울기 (변화량)


#linear regression
model = LinearRegression().fit(x.reshape(-1,1), y)
model.coef_
model.intercept_

#r2 score
model.score(x.reshape(-1,1), y)

#visualization
a, b = model.coef_[0], model.intercept_
plt.scatter(x, y)
plt.plot(x, a*x+b, linestyle='--', color='red' )
plt.grid()
plt.show()

#It's possible to make 'predictions' using this model
a*60 + b

model.predict([[60]])

#Multiple Regression

#sample
np.random.seed(12)
x1 = np.random.randn(800) * 70
x2 = np.random.randn(800) * 70
y = 7*x1 + 2*x2 - 90 + np.random.randn(800) * 70

#visualization for understanding
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(projection='3d')
ax.scatter(x1, x2, y, marker='x', color='b', alpha=0.2)
plt.grid()
plt.show()

x = np.c_[x1, x2]
x.shape

#beta1x1 + beta2x2 + b

model2d = LinearRegression().fit(x, y)
model2d.coef_
model2d.intercept_
model2d.score(x, y)

#visualization
fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(projection='3d')
ax.scatter(x1, x2, y, marker=',', color='green', alpha=0.1)

xx1 = np.tile(np.arange(-150, 150), (300,1))
xx2 = np.tile(np.arange(-150, 150), (300,1)).T

beta1 = model2d.coef_[0]
beta2 = model2d.coef_[1]
b = model2d.intercept_
yy = beta1 * xx1 + beta2 * xx2 + b
ax.plot_surface(xx1, xx2, yy, alpha=0.3)

plt.grid()
plt.show()


#polynomial regression
#non-linear model


#quadratic model
#f(x) = beta1x1 + beta2x1**2 + b
x = np.linspace(-1, 1, 800)
y = 5*x + 10*x**2 -5
plt.plot(x, y)
plt.grid()
plt.show()

#Qubic model
x = np.linspace(-1, 1, 800)
y = 5*x + 10*x**2 + 20*x**3 - 5
plt.plot(x, y)
plt.grid()
plt.show()

#Exponential model
x = np.linspace(-1, 1, 800)
y = np.exp(10*x)

plt.figure(figsize=(10, 5))
plt.subplot(1,2,1)
plt.plot(x,y)
plt.grid()

plt.subplot(1,2,2)
plt.plot(x, np.log(y))
plt.grid()
plt.show()



#random sample generate
np.random.seed(1234)
x_train = np.linspace(0, 1, 80)
y_train = np.sin(1.8 * np.pi * x_train) + (np.random.randn(80)/10)

x_test = np.linspace(0, 1, 20)
y_test = np.sin(1.8 * np.pi * x_test) + (np.random.randn(20)/10)

#visualization
plt.plot(x_train, y_train, 'o', label='train')
plt.plot(x_test, y_test, 'o', label='test')
plt.legend()
plt.grid()
plt.show()



x_train.shape
x_train.reshape(-1,1).shape


x_train = x_train.reshape(-1,1)
x_test = x_test.reshape(-1,1)

linearModel = LinearRegression().fit(x_train, y_train)
linearModel.coef_
linearModel.intercept_
linearModel.score(x_train, y_train)
linearModel.score(x_test, y_test)


plt.plot(x_train, y_train, 'o', label='train')
plt.plot(x_test, y_test, 'x', label='test')
#plt.plot(x_train, linearModel.predict(x_train))
plt.plot(x_train, linearModel.coef_[0] * x_train + linearModel.intercept_)
plt.legend()
plt.grid()
plt.show()




#quadratic model

x_quadratic = np.c_[x, x**2]
x_quadratic.shape
x_train_quadratic = np.c_[x_train, x_train**2]
x_test_quadratic = np.c_[x_test, x_test**2]

quadraticModel = LinearRegression().fit(x_train_quadratic, y_train)

quadraticModel.coef_
quadraticModel.intercept_
quadraticModel.score(x_train_quadratic, y_train)
quadraticModel.score(x_test_quadratic, y_test)


#visualization
plt.plot(x_train, y_train, 'o', label='train')
plt.plot(x_test, y_test, 'x', label='test')
plt.plot(x_train, quadraticModel.predict(x_train_quadratic))
plt.grid()
plt.legend()
plt.show()




#Qubic model
x_qubic = np.c_[x, x**2, x**3]
x_train_qubic = np.c_[x_train, x_train**2, x_train**3]
x_test_qubic = np.c_[x_test, x_test**2, x_test**3]

qubicModel = LinearRegression().fit(x_train_qubic, y_train)
qubicModel.coef_
qubicModel.intercept_
qubicModel.score(x_train_qubic, y_train)
qubicModel.score(x_test_qubic, y_test)

#visualization
plt.plot(x_train, y_train, 'o', label='train')
plt.plot(x_test, y_test, 'x', label='test')
plt.plot(x_train, qubicModel.predict(x_train_qubic))
plt.grid()
plt.legend()
plt.show()



#overfit
#sample generation
np.random.seed(1234)
x = np.linspace(0, 1, 1000)
y = np.sin(2 * np.pi * x) + (np.random.randn(1000)/5)

x_train = np.linspace(0, 1, 11)
y_train = np.sin(2 * np.pi * x_train) + (np.random.randn(11)/5)

x_test = np.linspace(0, 1, 50)
y_test = np.sin(2 * np.pi * x_test) + (np.random.randn(50)/5)

x_poly = np.c_[ 
  x, x ** 2, x ** 3, x ** 4, x ** 5, x ** 6, x ** 7, x ** 8, x ** 9, x**10
]

x_train_poly = np.c_[ 
  x_train, x_train ** 2, x_train ** 3, x_train ** 4, x_train ** 5, x_train ** 6, x_train ** 7, x_train ** 8, x_train ** 9, x_train**10
]

x_test_poly = np.c_[ 
  x_test, x_test ** 2, x_test ** 3, x_test ** 4, x_test ** 5, x_test ** 6, x_test ** 7, x_test ** 8, x_test ** 9, x_test**10
]
polyModel = LinearRegression().fit(x_train_poly, y_train)
polyModel.score(x_train_poly, y_train)
polyModel.score(x_test_poly, y_test)
plt.plot(x_train, y_train, 'o', label='train')
plt.plot(x_test, y_test, 'o', label='test')
plt.plot(x, polyModel.predict(x_poly))
plt.legend()
plt.grid()
plt.show()