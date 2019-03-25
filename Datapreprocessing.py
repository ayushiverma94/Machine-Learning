#Data Preprocessing

#Importing the libraries
import pandas as pd 
import numpy as np 

#Importing the csv file
df=pd.read_csv("D:/AYUSHI VERMA/Trainings/MACHINE LEARNING/Machine Learning A-Z/Part 1 - Data Preprocessing/Data.csv")
print(df)  #printing Original Dataset

print("  ")

#Handling missing data
	#Taking care of missing values in age column
k=df['Age'].mean()
df['Age']=df['Age'].fillna(k)

	#Taking care of missing values in salary column
z=df['Salary'].mean()
df['Salary']=df['Salary'].fillna(z)

#Encoding the data - Taking care of Categorical data
	#Creating Dummy Variables for country column
x=pd.get_dummies(df.Country)
df=pd.concat([df,x], axis=1)
#df=df.drop(['Country'],axis=1)  


	#Creating Dummy Variables for purchased column
df['purchased']=df.Purchased.map({'No':0,'Yes':1})


#Splitting the dataset into Training set and test set
from sklearn.model_selection import train_test_split
a=df[['Age','Salary']]    # Independent Variables 
b=df['purchased']		  # Dependent Variables
a_train,a_test,b_train,b_test = train_test_split(a,b,test_size = 0.2, random_state = 0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
standard_a=StandardScaler()
a_train=standard_a.fit_transform(a_train)
a_test=standard_a.fit_transform(a_test)
print(df)  # Printing datasets with dummy variablesz
print("  ")	
print(a_train)
print("  ")	
print(a_test)

#to write and create new csv file
df.to_csv("D:/AYUSHI VERMA/Trainings/MACHINE LEARNING/Machine Learning A-Z/Part 1 - Data Preprocessing/Data_updated.csv")
