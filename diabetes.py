from pyexpat import model
from statistics import mode
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error

db = datasets.load_diabetes()

# print(db.keys())
# print(db.DESCR) 
# dict_keys(['data', 'target', 'frame', 'DESCR', 'feature_names', 'data_filename', 'target_filename', 'data_module'])

db_x = db.data

# print(db_x)

db_x_train = db_x[:-20]
db_x_test = db_x[-20:]

db_y_train = db.target[:-20]
db_y_test = db.target[-20:]

model = linear_model.LinearRegression()
model.fit(db_x_train, db_y_train)

db_y_predicted = model.predict(db_x_test)

print("MSE: ", mean_squared_error(db_y_test, db_y_predicted))
print("W: ", model.coef_)
print("slope: ", model.intercept_)

# plt.scatter(db_x_test, db_y_test)
# plt.plot(db_x_test, db_y_predicted)

plt.show()

