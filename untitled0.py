# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 21:26:40 2021

@author: HP
"""
#Import Libraries
import pickle as pkl
import pandas as pd
import numpy as np
import seaborn as sns
#pip install imbalanced-learn
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix,accuracy_score,cohen_kappa_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from scipy.special import boxcox1p
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt



import pandas as pd

dataset = pd.read_csv('data_cart_abandonment.csv')
dataset.head(3)
dataset.dtypes
dataset.Cart_Abandoned=pd.Categorical(dataset.Cart_Abandoned)
dataset.Cart_Abandoned.dtypes
dataset.Customer_Segment_Type=pd.Categorical(dataset.Customer_Segment_Type)
dataset.dtypes


#EDA
#Univariate
sns.countplot(dataset.Cart_Abandoned)
num=dataset.select_dtypes(include=["float64","int64"])
cat=dataset.select_dtypes(include=["object","category"]).drop(["ID"],axis=1)
num.shape
num.hist(bins=15, figsize=(20, 6), layout=(2, 5));
f, ax = plt.subplots(nrows=1,ncols=3,figsize=(20, 8))
for i,j in zip(cat.columns.tolist(), ax.flatten()):
    sns.countplot(x=cat[i],ax=j)
    
#BiVariate
#1.FOR CATEGORICAL :Is_Product_Details_viewed Vrs Cart_Abandoned
#Part 1 : If a customer is viewing the product details then what is the chance that he is doing cart abandonment?

sns.countplot(x=dataset.Is_Product_Details_viewed,hue=dataset.Cart_Abandoned)

#2.1 FOR NUMERICAL : Numerical Attributes Vrs Cart_Abandoned : By BoxPlot
fig, ax = plt.subplots(2, 5, figsize=(20, 10))
for var, subplot in zip(num.columns.tolist(), ax.flatten()):
    sns.boxplot(x=cat["Cart_Abandoned"], y=num[var], ax=subplot)
    
#2.2 Correlation plot of Independent attributes
corr = num.corr()
sns.heatmap(corr)

#3. Data Preparation
#imputing missing values wherever needed
data = dataset.copy()
data.isna().sum()
data['No_Cart_Viewed'].median()
data['No_Items_Added_InCart'].median()
data['No_Items_Added_InCart'].median()
data['No_Items_Added_InCart'].fillna(2.0,inplace=True)
data['No_Cart_Viewed'].median()
data['No_Cart_Viewed'].fillna(1.0,inplace=True)
data.columns
data.dtypes
data['No_Items_Added_InCart']= data['No_Items_Added_InCart'].astype(int)
data['No_Cart_Viewed']= data['No_Cart_Viewed'].astype(int)


#1.Taking care of Outliers by Normalizing the Data : By MinMax Normalization
num=data.select_dtypes(include=["int64","int32"])
cat=data.select_dtypes(include=["object","category"]).drop(["ID"],axis=1)
min_max_scaler = MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(num)
x_scaled
df_scaled = pd.DataFrame(x_scaled,columns=num.columns)
df_scaled.head(3)
df_scaled.dtypes
df_scaled.hist(bins=15, figsize=(20, 6), layout=(2, 5));

fig, ax = plt.subplots(2, 5, figsize=(20, 10))
for var, subplot in zip(df_scaled.columns.tolist(), ax.flatten()):
    sns.boxplot(y=df_scaled[var], ax=subplot)

#2.Taking care of Outliers by Normalizing the Data : By BoxCox Normalization
df_scaled_boxcox=boxcox1p(num, 0)
fig, ax = plt.subplots(2, 5, figsize=(20, 10))
for var, subplot in zip(df_scaled_boxcox.columns.tolist(), ax.flatten()):
    sns.boxplot(y=df_scaled_boxcox[var], ax=subplot)

df_scaled_boxcox.hist(bins=15, figsize=(20, 6), layout=(2, 5));

df_scaled_boxcox.head(3)
df_scaled_boxcox.dtypes
df_scaled_boxcox["ID"]=data.ID
df_scaled_boxcox.set_index('ID',inplace=True)
df_scaled_boxcox.reset_index(inplace=True)
df_final=df_scaled_boxcox.join(cat)
df_final.head(3)
df_final.Is_Product_Details_viewed.replace({"Yes":1,"No":0},inplace=True)
df_final.head(3)
df_final.dtypes

df_final.Is_Product_Details_viewed=pd.Categorical(df_final.Is_Product_Details_viewed)
df_final['Session_Activity_Count']= df_final['Session_Activity_Count'].astype(int)
df_final['No_Items_Added_InCart']= df_final['No_Items_Added_InCart'].astype(int)
df_final['No_Items_Removed_FromCart']= df_final['No_Items_Removed_FromCart'].astype(int)
df_final['No_Cart_Viewed']= df_final['No_Cart_Viewed'].astype(int)
df_final['No_Checkout_Confirmed']= df_final['No_Checkout_Confirmed'].astype(int)
df_final['No_Checkout_Initiated ']= df_final['No_Checkout_Initiated '].astype(int)

df_final['No_Cart_Items_Viewed']= df_final['No_Cart_Items_Viewed'].astype(int)
df_final['No_Customer_Login']= df_final['No_Customer_Login'].astype(int)
df_final['No_Page_Viewed']= df_final['No_Page_Viewed'].astype(int)
df_final['Is_Product_Details_viewed']= df_final['Is_Product_Details_viewed'].astype(int)
df_final['Customer_Segment_Type']= df_final['Customer_Segment_Type'].astype(int)
df_final['Cart_Abandoned']= df_final['Cart_Abandoned'].astype(int)






#Feature Selection
#1. By RFE
X=df_final.iloc[:,1:12]
X.shape
y=df_final["Cart_Abandoned"]
y.name

lr = LogisticRegression()
lr.fit(X,y)

#stop the search when only the last feature is left

rfe = RFE(lr, n_features_to_select=7, verbose = 3 )
fit=rfe.fit(X,y)

print("Num Features: %d"% fit.n_features_) 
print("Selected Features: %s"% fit.support_) 
print("Feature Ranking: %s"% fit.ranking_)

l = [i for i,x in enumerate(list(fit.support_)) if x == True]

X.columns

feature_selected = [X[X.columns[l[i]]].name for i,x in enumerate(l)]
feature_selected



#Part 2: What are the important factors related to cart abandonment?


#Train-Validation Split
X = df_final.iloc[:,[1,4,5,6,8,9,10]]
X.dtypes
y = df_final.loc[:,["Cart_Abandoned"]]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state=0)
#Over-Sampling Using SMOTE
sm = SMOTE(random_state=2,k_neighbors=5)

#X_train, y_train =sm.fit_sample(X_train,y_train)
X_train, y_train =sm.fit_resample(X_train,y_train)

#Train-Validation Split after SMOTE
X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(X_train, y_train, test_size=0.40, random_state=0)
#4. Model Builiding And Prediction
df_final.dtypes
#Part 3: How can we able to know whether a customer will do abandonment or not in future?
lr1 = LogisticRegression()
lr1.fit(X_train_new,y_train_new)
X_train_new.dtypes
y_train_new.dtypes

y_pred_new = lr1.predict(X_test_new)  #### For SMOTE validation samples
y_pred=lr1.predict(X_test)##### For actual validation samples


#5. Evaluation
print(" accuracy is %2.3f" % accuracy_score(y_test_new, y_pred_new))
print(" Kappa is %f" %cohen_kappa_score(y_test_new, y_pred_new))
print(" accuracy is %2.3f" % accuracy_score(y_test, y_pred))
print(" Kappa is %f" %cohen_kappa_score(y_test, y_pred))

import pickle
pickle.dump(regressor, open('model.pkl','wb'))
