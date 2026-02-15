import pandas as pd
from sklearn.linear_model import LinearRegression

data = pd.read_csv("student_data.csv")
features = ['Hours_Studied', 'Attendance', 'Previous_Score']
X = data[features]
y = data['Final_Score']

model = LinearRegression()
model.fit(X, y)

hours = float(input("Enter hours studied: "))
attendance = float(input("Enter attendance %: "))
previous_score = float(input("Enter previous score: "))

predicted_score = model.predict([[hours, attendance, previous_score]])[0]
print(f"Predicted Final Score: {predicted_score:.2f}")
