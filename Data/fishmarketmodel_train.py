import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
#from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import joblib
data = pd.read_csv('train/fish_weekly_prices.csv')
data['date'] = pd.to_datetime(data['date'],dayfirst=True)
data['date'] = (data['date'] - pd.to_datetime('22/3/2024',dayfirst=True)) / pd.Timedelta(days=1)
data=pd.get_dummies(data)
X = data.drop('avg_price_kg',axis=1)
y = data['avg_price_kg']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
'''scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)'''
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print("Test:",r2_score(y_test, predictions))
predictions = model.predict(X_train)
print("Train:",r2_score(y_train, predictions))
joblib.dump(model, 'models/fishprice.pkl')
