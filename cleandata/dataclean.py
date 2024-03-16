import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt

# Read the CSV file into a pandas DataFrame
file_path = r'C:\Users\Yan Zhang\OneDrive - National University of Singapore\Desktop\Y3S1\libraryfrontend\resale_flat.csv'
data = pd.read_csv(file_path)

# Convert 'flat_type' and 'flat_model' columns into numerical data
le_flat_type = LabelEncoder()
data['flat_type'] = le_flat_type.fit_transform(data['flat_type'])

le_flat_model = LabelEncoder()
data['flat_model'] = le_flat_model.fit_transform(data['flat_model'])

# Split the dataset into training and testing sets
X = data[['flat_type', 'flat_model', 'floor_area_sqm']]  
y = data['resale_price']  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scatter plot between 'floor_area_sqm' and 'resale_price'
sns.scatterplot(x='floor_area_sqm', y='resale_price', data=data)
plt.title('Scatter plot of Floor Area vs Resale Price')
plt.xlabel('Floor Area (sqm)')
plt.ylabel('Resale Price (SGD)')
plt.show()

# Scatter plot between 'flat_type' and 'resale_price'
sns.scatterplot(x='flat_type', y='resale_price', data=data)
plt.title('Scatter plot of Flat Type vs Resale Price')
plt.xlabel('Flat Type')
plt.ylabel('Resale Price (SGD)')
plt.show()

# Scatter plot between 'flat_model' and 'resale_price'
sns.scatterplot(x='flat_model', y='resale_price', data=data)
plt.title('Scatter plot of Flat Model vs Resale Price')
plt.xlabel('Flat Model')
plt.ylabel('Resale Price (SGD)')
plt.show()

# Correlation matrix
corr_matrix = data[['flat_type', 'flat_model', 'floor_area_sqm', 'resale_price']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()
