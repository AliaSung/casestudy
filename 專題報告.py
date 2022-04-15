import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


#資料匯入
x = pd.read_csv("C:\\Users\\alia\\Desktop\\cpii.csv")
target = pd.read_csv("C:\\Users\\alia\\Desktop\\datamon.csv")
y = target
data=pd.read_csv("C:\\Users\\alia\\Desktop\\data.csv")

#相關係數
print("相關係數:")
print(data.corr())

#CPI圖
plt.scatter(x.cpii, y)
plt.xlabel("CPI")
plt.ylabel("CriminalCases")
# plt.title("CPI VS CriminalCases" )
plt.show()

#破案圖
plt.scatter(x.cdR, y)
plt.xlabel("Case Detection Rate")
plt.ylabel("CriminalCases")
plt.title("Case Detection Rate VS CriminalCases" )
plt.show()

#景氣圖
plt.scatter(x.boom, y)
plt.xlabel("Prosperity Index")
plt.ylabel("CriminalCases")
plt.title("Prosperity Index VS CriminalCases" )
plt.show()

#失業率圖
plt.scatter(x.ueR, y)
plt.xlabel("Uneployment Rate")
plt.ylabel("CriminalCases")
plt.title("Uneployment Rate VS CriminalCases" )
plt.show()

#工業及服務業薪水圖
plt.scatter(x.salary, y)
plt.xlabel("Salary")
plt.ylabel("CriminalCases")
plt.title("Salary VS CriminalCases" )
plt.show()


#圓餅圖
#匯入資料
x = [" 竊盜","其他刑案","暴力犯罪"]
y =[5775813,5205489,227925]
colors = ["cornflowerblue","lightsteelblue","mediumblue"]
explode=[0,0.05,0.2]

#中文基本設定
plt.rcParams["font.family"]='Microsoft Yahei'
plt.rcParams["font.size"]=12

font_path="C:\Windows\Fonts\msjh.ttc"
font_prop=fm.FontProperties(fname=font_path)
font_prop.set_style("normal")
font_prop.set_size("12")

#圓餅圖
#設定圖表類及相關設定
plt.pie(y, labels = x, colors=colors,shadow=True,
        explode = explode,autopct="%1.1f%%")
plt.axis("equal")
plt.show()