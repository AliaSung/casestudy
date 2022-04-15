import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

x = pd.read_csv("C:\\Users\\alia\\Desktop\\break.csv")
target = pd.read_csv("C:\\Users\\alia\\Desktop\\datamon.csv")
y = target

#破案率迴歸係數\MSE\R-squared
lm = LinearRegression()
lm.fit(x,y)
print("回歸:",lm.coef_)
predicted_case = lm.predict(x)
MSE_x = np.mean((y-predicted_case)**2)
print("MSE:", MSE_x)
print("R-squared:", lm.score(x, y))

plt.scatter(x, y)
plt.xlabel("Case Detection Rate")
plt.ylabel("CriminalCases")
plt.title("Case Detection Rate VS CriminalCases" )
plt.show()

plt.scatter(y, predicted_case)
plt.xlabel("target")
plt.ylabel("predicted_target")
plt.show()
