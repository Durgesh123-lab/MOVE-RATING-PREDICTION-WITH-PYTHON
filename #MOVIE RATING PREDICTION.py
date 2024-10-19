# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Sample dataset creation (for simplicity)
data = {'Genre': ['Action', 'Comedy', 'Drama', 'Action', 'Comedy'],
        'Director': ['Director A', 'Director B', 'Director A', 'Director C', 'Director B'],
        'Actors': ['Actor 1, Actor 2', 'Actor 3, Actor 4', 'Actor 1, Actor 3', 'Actor 4, Actor 2', 'Actor 1, Actor 5'],
        'Budget': [100, 50, 70, 120, 60],
        'Rating': [7.5, 6.0, 8.0, 7.0, 6.5]}

# Converting dataset to DataFrame
df = pd.DataFrame(data)

# Encoding categorical features (Genre, Director, and Actors)
label_encoder = LabelEncoder()

df['Genre'] = label_encoder.fit_transform(df['Genre'])
df['Director'] = label_encoder.fit_transform(df['Director'])
df['Actors'] = label_encoder.fit_transform(df['Actors'])

# Features (X) and Target (y)
X = df[['Genre', 'Director', 'Actors', 'Budget']]
y = df['Rating']

# Splitting the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model creation and training
model = LinearRegression()
model.fit(X_train, y_train)

# Making predictions
y_pred = model.predict(X_test)

# Calculating the Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)

print("Predicted Ratings:", y_pred)
print("Mean Squared Error:", mse)
