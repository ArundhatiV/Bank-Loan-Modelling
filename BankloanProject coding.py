# -*- coding: utf-8 -*-
"""
Created on Wed May 18 23:06:35 2022

@author: Admin
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_excel('C:/Users/Admin/Desktop/Project prep/Group 3 - Financial Analytics - Bank Loan Modelling/Project-Bank_Personal_Loan_Modelling.xlsx',sheet_name=2)
data.head()
data.columns


from sklearn import preprocessing 
le=preprocessing.LabelEncoder()
data['Securities Account']=le.fit_transform(data['Securities Account'])
data['CD Account']=le.fit_transform(data['CD Account'])
data['Online']=le.fit_transform(data['Online'])
data['CreditCard']=le.fit_transform(data['CreditCard'])
data['Personal Loan']=le.fit_transform(data['Personal Loan'])

data.isna().sum()
data.drop_duplicates()


data.describe().T
data.skew()
data.kurt()
data.corr()

plt.hist(data.Income)
plt.boxplot(data.Income)

data_no=data[data['Personal Loan']==0]
data_yes=data[data['Personal Loan']==1]


from scipy.stats import mannwhitneyu
stats,p=mannwhitneyu(data_no.Income,data_yes.Income)
print(stats,p)

# Income has significantly affecting personal loan.

plt.hist(data.Education)
plt.boxplot(data.Education)

from scipy.stats import chi2_contingency
chitable=pd.crosstab(data.Education,data['Personal Loan'])
stats,p,dof,expected=chi2_contingency(chitable)
print(stats,p)

plt.hist(data.Family)
plt.boxplot(data.Family)


from scipy.stats import chi2_contingency
chitable=pd.crosstab(data.Family,data['Personal Loan'])
stats,p,dof,expected=chi2_contingency(chitable)
print(stats,p)

plt.hist(data.CCAvg)
stats,p=mannwhitneyu(data_no.CCAvg,data_yes.CCAvg)
print(stats,p)

plt.hist(data.Experience)

from scipy.stats import ttest_ind
stats,p=ttest_ind(data_no.Experience,data_yes.Experience)
print(stats,p)

plt.hist(data.Age)
stats,p=ttest_ind(data_no.Age,data_yes.Age)
print(stats,p)




#MOdelling- Logistic REgression


import statsmodels.api as sm
y=data['Personal Loan']
x=data.drop(['Personal Loan'],axis=1)
x1=sm.add_constant(x)
log=sm.Logit(y,x1)
result=log.fit()
result.summary()
