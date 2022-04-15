import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.formula.api import ols

x = pd.read_csv("C:\\Users\\alia\\Desktop\\all.csv")
target = pd.read_csv("C:\\Users\\alia\\Desktop\\datamon.csv")
y = target

#景氣,CPI,破案率複迴歸
lm = LinearRegression()
lm.fit(x,y)
print("回歸:",lm.coef_)
predicted_all = lm.predict(x)
MSE_x = np.mean((y-predicted_all)**2)
print("MSE:", MSE_x)
print("R-squared:", lm.score(x, y))


#T型分析，求p值
data2 = pd.read_csv("C:\\Users\\alia\\Desktop\\sas.csv")
x1=np.array(data2["cpi"])
x2=np.array(data2["boom"])
x3=np.array(data2["break"])
y1=np.array(data2["case"])
model=ols("y1 ~ x1 + x2 + x3",data=data2).fit()
print(model.summary())

#出圖
plt.scatter(y, predicted_all)
plt.xlabel("target")
plt.ylabel("predicted_target")
plt.show()