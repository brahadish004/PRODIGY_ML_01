import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
from sklearn.model_selection import cross_val_score

train_data = pd.read_csv('/content/train.csv')
test_data = pd.read_csv('/content/test.csv')

print(train_data.info())
print(train_data.describe())

train_data.fillna(0, inplace=True)
test_data.fillna(0, inplace=True)

X_train =  train_data.drop('SalePrice', axis=1)
y_train =  train_data['SalePrice']

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.20,random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_val)

mae = mean_absolute_error(y_val, y_pred)
mse = mean_squared_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)

print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("R-squared:", r2)

X_test = test_data
test_predictions = model.predict(X_test)

submission = pd.DataFrame({'Id': test_data['Id'], 'Predicted_Salesprice': test_predictions})
submission.to_csv('PredictedPrice.csv', index=False)