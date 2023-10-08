from model import neural_network, calculate_test_accuracy
from sklearn.model_selection import train_test_split
import pandas as pd

data = pd.read_csv('framingham.csv')

X = data.iloc[:, :-1]  
y = data.iloc[:, -1]   

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
