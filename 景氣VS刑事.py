import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

x = pd.read_csv("C:\\Users\\alia\\Desktop\\boom.csv")
target = pd.read_csv("C:\\Users\\alia\\Desktop\\datamon.csv")
y = target

#景氣迴歸係數\MSE\R-squared
lm = LinearRegression()
lm.fit(x,y)
print("回歸:",lm.coef_)
predicted_boom = lm.predict(x)
MSE_x = np.mean((y-predicted_boom)**2)
print("MSE:", MSE_x)
print("R-squared:", lm.score(x, y))

plt.scatter(x.num, y)
plt.xlabel("Prosperity Index")
plt.ylabel("CriminalCases")
plt.title("Prosperity Index VS CriminalCases" )
plt.show()

plt.scatter(y, predicted_boom)
plt.xlabel("target")
plt.ylabel("predicted_target")
plt.show()