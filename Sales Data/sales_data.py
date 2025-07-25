import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
df=pd.read_excel(r'C:\Users\hites\Downloads\MLResearch\MLResearch\Sales Prediction Dataset\advertising_sales_data.xlsx')
df.drop(columns=['Campaign'],inplace=True)
print(df.info())
#checking for missing values
print(df.isnull().sum())

#filling the missing values in 'Radio' Coloumn
sns.boxplot(y='Radio',data=df)
plt.title('Box plot for Radio')
plt.show()

df['Radio']=df['Radio'].fillna(df['Radio'].mean())
print(df.info())
print(df.describe())


#1.What is the average amount spent on TV advertising in the dataset?
average_tv=df['TV'].mean()
print('The average amount spent on TV advertising is',average_tv)

#2.What is the correlation between radio advertising expenditure and product sales?
correlation =df[['Radio','Sales']].corr()
print(correlation)
sns.heatmap(correlation, annot=True, cmap='PuBu')
plt.title('correlation between radio advertising expenditure and product sales')
plt.show()

#3.Which advertising medium has the highest impact on sales based on the dataset?
correlation =df[['TV','Radio','Newspaper','Sales']].corr()
print(correlation)
sns.heatmap(correlation, annot=True, cmap='PuBuGn')
plt.title('Correlation between TV,Newspaper,Radio advertising expenditure and product sales')
plt.show()

#4.Plot a linear regression line that includes all variables (TV, Radio, Newspaper) to predict Sales, and visualize the model's predictions against the actual sales values.
tv = df.iloc[:,0].values
print(tv)
radio = df.iloc[:,1].values
print(radio)
newspaper = df.iloc[:,2].values
print(newspaper)
sales = df.iloc[:,3].values
print(sales)

x = np.column_stack((tv, radio, newspaper))
y = sales

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)
print(y_pred)

plt.scatter(y_test, y_pred, color='blue')

plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('Actual vs Predicted Sales')

#5.How would sales be predicted for a new set of advertising expenditures: $200 on TV, $40 on Radio, and $50 on Newspaper?
new_data = np.array([[200, 40, 50]])  # 2D array with shape (1, 3)
predicted_sales = regressor.predict(new_data)
print("Predicted Sales:", predicted_sales)

#6.How does the performance of the linear regression model change when the dataset is normalized?(it stays the same)
sc = StandardScaler()
x_train_sc = sc.fit_transform(x_train)
x_test_sc = sc.transform(x_test)

regressor_sc = LinearRegression()
regressor_sc.fit(x_train_sc, y_train)
y_pred_sc = regressor_sc.predict(x_test_sc)

plt.scatter(y_test, y_pred_sc, color='blue', label='Predicted Sales')
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('Actual vs Predicted Sales (StandardScaler)')
plt.show()

#7.What is the impact on the sales prediction when only radio and newspaper advertising expenditures are used as predictors?
x = np.column_stack((radio, newspaper))
y = sales

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)
print(y_pred)

plt.scatter(y_test, y_pred, color='blue')

plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('Actual vs Predicted Sales(TV excluded)')


