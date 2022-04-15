import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

x = pd.read_csv("C:\\Users\\alia\\Desktop\\CPI.csv")
target = pd.read_csv("C:\\Users\\alia\\Desktop\\datamon.csv")
y = target

#CPI回歸\MSE\R-squared
lm = LinearRegression()
lm.fit(x,y)
print("回歸:",lm.coef_)
predicted_cpi = lm.predict(x)
MSE_x = np.mean((y-predicted_cpi)**2)
print("MSE:", MSE_x)
print("R-squared:", lm.score(x, y))


plt.scatter(x, y)
plt.xlabel("CPI")
plt.ylabel("CriminalCases")
plt.title("CPI VS CriminalCases" )
plt.show()

plt.scatter(y, predicted_cpi)
plt.xlabel("target")
plt.ylabel("predicted_target")
plt.show()