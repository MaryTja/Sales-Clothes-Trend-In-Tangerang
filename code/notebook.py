#!/usr/bin/env python
# coding: utf-8

# <div> Nama: Mary Elizabeth Tjang 
# <div> NIM: 00000057284 
# <div> Mata Kuliah: IS388 - Data Analysis 

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 


# In[2]:


sales = pd.read_csv("sales.csv")
sales.head()


# In[3]:


sales.info()


# In[4]:


sales.describe()


# In[5]:


sales['total'] = sales['qty_ordered'] * sales['price']


# In[6]:


# i. Distribution - Histogram
plt.figure(figsize=(10, 6))
plt.hist(sales['total'], bins=20, color='skyblue', edgecolor='black')
plt.title('total')
plt.xlabel('total')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()


# In[15]:


#Relationship

numeric_columns = ["total", "qty_ordered", "price"]

correlation_matrix = sales[numeric_columns].corr()

# Heatmap
plt.figure(figsize=(8, 6))  # Set the figure size (optional)
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", linewidths=0.5)

plt.title("Heatmap")

plt.show()


# In[16]:


#Comparison

#Line Chart
sales["total"] = sales["qty_ordered"] * sales["price"]

sales["order_date"] = pd.to_datetime(sales["order_date"])

daily_sales = sales.groupby("order_date")["total"].sum().reset_index()

plt.figure(figsize=(12, 6))  
plt.plot(daily_sales["order_date"], daily_sales["total"])

plt.xlabel("order_date")
plt.ylabel("total")
plt.title(" daily total sales")

plt.show()


# In[8]:


#c. 
# Distribution 
#Density Plot 

column_name = "total"
selected_column = sales[column_name]

# Density plot
plt.figure(figsize=(10, 6))  
sns.kdeplot(selected_column, shade=True)

plt.xlabel(column_name)
plt.ylabel("Density")
plt.title(f"Density Plot of {column_name}")

plt.show()


# In[9]:


payment_method_counts = sales['payment_method'].value_counts()
others=payment_method_counts<10000
other_count = payment_method_counts[others].sum()
payment_method_counts[others]
payment_method_counts = payment_method_counts[~others]
payment_method_counts['Other'] = other_count

# Pie Chart 
plt.figure(figsize=(8, 6))
plt.pie(payment_method_counts, labels=payment_method_counts.index, autopct='%1.1f%%')

plt.title('Distribution of Payment Methods')

plt.show()


# In[19]:


#Bar plot 
colors = sns.color_palette('pastel')[3]
sns.histplot(data=sales, x="order_date", color=colors)


# In[11]:


#Distribution Plot 
sns.displot(sales, x="year", hue="qty_ordered", kind="kde", multiple="stack")


# In[12]:


#scatter plot 
sns.scatterplot(data=sales, x='year', y="total", palette='brown')


# In[13]:


#Heatmap 
sns.heatmap(sales[['year', 'qty_ordered', 'total']].corr(), annot=True)


# In[14]:


#Binding data
bins = [0, 250, 350, 500]
labels = ['Low', 'Medium', 'High']
sales['total_Binned'] = pd.cut(sales['total'], bins=bins, labels=labels, right=False)
print(sales)


# In[15]:


#Normalize data
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
sales['total'] = scaler.fit_transform(sales[['year']])
print(sales)


# HANDLING OUTLIERS

# In[60]:


# Handling outliers (Example: Removing outliers)
sales = sales[~((sales['price'] < (Q1 - 1.5 * IQR)) | (sales['price'] > (Q3 + 1.5 * IQR)))]


# In[59]:


# Identifying outliers (Example using IQR for the 'price' column)
Q1 = sales['price'].quantile(0.25)
Q3 = sales['price'].quantile(0.75)
IQR = Q3 - Q1
outliers = sales[(sales['price'] < (Q1 - 1.5 * IQR)) | (sales['price'] > (Q3 + 1.5 * IQR))]


# FORMATTING

# In[61]:


# Convert 'order_date' to datetime format
sales['order_date'] = pd.to_datetime(sales['order_date'])

# Ensure numerical columns are the correct type
sales['qty_ordered'] = sales['qty_ordered'].astype(float)


# NORMALIZATION

# In[63]:


# Normalizing a column (Example: Min-Max Scaling for 'price')
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

sales['price_normalized'] = scaler.fit_transform(sales[['price']])


# ENCODING

# In[68]:


# Encoding categorical data (Example: One-Hot Encoding for 'status')
status_encoded = pd.get_dummies(sales['status'], prefix='status')
sales = sales.join(status_encoded)


# BINNING

# In[65]:


# Binning 'price' into categories
price_bins = [0, 50, 100, 200, 500, 1000, 5000]
price_labels = ['0-50', '51-100', '101-200', '201-500', '501-1000', '1001-5000']
sales['price_bin'] = pd.cut(sales['price'], bins=price_bins, labels=price_labels)


# GROUPING

# In[69]:


# Grouping data (Example: Average price by City)
average_price_by_city = sales.groupby('City')['price'].mean()


# GENDER CLASSIFICATION

# In[17]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load the dataset
sales = pd.read_csv('sales.csv')

# Handle missing values
# Assuming 'gender' is the target column and it should not have missing values
sales.dropna(subset=['Gender'], inplace=True)

# Select relevant features
# This selection depends on your domain knowledge and understanding of the dataset
features = ['qty_ordered', 'price', 'City', 'State', 'Region']  # example features
X = sales[features]

# Encode categorical data
label_encoder = LabelEncoder()
for feature in ['City', 'State', 'Region']:
    X[feature] = label_encoder.fit_transform(X[feature])

# Define the target variable
y = label_encoder.fit_transform(sales['Gender'])

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[18]:


from sklearn.linear_model import LogisticRegression

# Initialize the Logistic Regression model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)


# In[84]:


from sklearn.metrics import classification_report, accuracy_score

# Predicting the Test set results
y_pred = model.predict(X_test)

# Generating the classification report and accuracy
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))


# REGION CLASSIFICATION

# In[20]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load the dataset
sales = pd.read_csv('sales.csv')

# Handle missing values
sales.dropna(inplace=True)

# Select relevant features
# Note: Adjust the feature list based on your dataset
features = ['qty_ordered', 'price', 'City', 'State', 'item_id', 'discount_amount']  # example features
X = sales[features]

# Encode categorical data
label_encoder = LabelEncoder()
for feature in X.select_dtypes(include=['object']).columns:
    X[feature] = label_encoder.fit_transform(X[feature])

# Define the target variable
y = label_encoder.fit_transform(sales['Region'])

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[21]:


from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()

model.fit(X_train, y_train)


# In[87]:


from sklearn.metrics import classification_report, accuracy_score

# Predicting the Test set results
y_pred = model.predict(X_test)

# Generating the classification report and accuracy
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))


# KNN CLASSIFIER

# In[23]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the dataset
sales = pd.read_csv('sales.csv')

# Handle missing values
sales.dropna(inplace=True)

# Select relevant features
# Adjust the feature list based on your dataset
features = ['qty_ordered', 'price', 'City', 'State', 'item_id', 'discount_amount']  # example features
X = sales[features]

# Encode categorical data
label_encoder = LabelEncoder()
for feature in X.select_dtypes(include=['object']).columns:
    X[feature] = label_encoder.fit_transform(X[feature])

# Define the target variable
y = label_encoder.fit_transform(sales['Region'])  # or any other categorical target variable

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Normalize the feature data - important for KNN
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[24]:


from sklearn.neighbors import KNeighborsClassifier

# Initialize the KNN classifier
# You can adjust the number of neighbors (n_neighbors)
knn_model = KNeighborsClassifier(n_neighbors=5)

# Train the model
knn_model.fit(X_train, y_train)


# In[91]:


from sklearn.metrics import classification_report, accuracy_score

# Predicting the test set results
y_pred = knn_model.predict(X_test)

# Generating the classification report and accuracy
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))


# In[18]:


X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)


# Random Forest Classifier

# In[27]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer


# In[28]:


# Load the dataset
sales = pd.read_csv('sales.csv')

# Handle missing values in the 'Region' column and any other crucial columns
sales.dropna(subset=['Region'], inplace=True)

# Select relevant features
features = ['qty_ordered', 'price', 'City', 'State', 'item_id', 'discount_amount']  # Example features
X = sales[features]

# Identify categorical columns for encoding
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

# Apply OneHotEncoder to categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), categorical_features)
    ], remainder='passthrough')

X_processed = preprocessor.fit_transform(X)

# Encode the target variable 'Region'
y = LabelEncoder().fit_transform(sales['Region'])

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.3, random_state=42)


# In[29]:


from sklearn.ensemble import RandomForestClassifier

# Initialize the Random Forest classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)  # n_estimators can be adjusted

# Train the model
rf_model.fit(X_train, y_train)


# In[4]:


from sklearn.metrics import classification_report, accuracy_score

# Predicting the test set results
y_pred = rf_model.predict(X_test)

# Generating the classification report and accuracy
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

