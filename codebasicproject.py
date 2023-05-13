# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 11:54:05 2023

@author: Sudeepthi 
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
##domestic visitors  2016-2019##
ds1=pd.read_csv("C:/Users/user/OneDrive/Desktop/Tourism/C5 Input for participants/domestic_visitors/domestic_visitors_2016.csv")
ds2=pd.read_csv("C:/Users/user/OneDrive/Desktop/Tourism/C5 Input for participants/domestic_visitors/domestic_visitors_2017.csv")
ds3=pd.read_csv("C:/Users/user/OneDrive/Desktop/Tourism/C5 Input for participants/domestic_visitors/domestic_visitors_2018.csv")
ds4=pd.read_csv("C:/Users/user/OneDrive/Desktop/Tourism/C5 Input for participants/domestic_visitors/domestic_visitors_2019.csv")
df=pd.concat([ds1,ds2,ds3,ds4])
## foreign visitors 2016-2019##
fv1=pd.read_csv("C:/Users/user/OneDrive/Desktop/Tourism/C5 Input for participants/foreign_visitors/foreign_visitors_2016.csv")
fv2=pd.read_csv("C:/Users/user/OneDrive/Desktop/Tourism/C5 Input for participants/foreign_visitors/foreign_visitors_2016.csv")
fv3=pd.read_csv("C:/Users/user/OneDrive/Desktop/Tourism/C5 Input for participants/foreign_visitors/foreign_visitors_2018.csv")
fv4=pd.read_csv("C:/Users/user/OneDrive/Desktop/Tourism/C5 Input for participants/foreign_visitors/foreign_visitors_2019.csv")
df1=pd.concat([fv1,fv2,fv3,fv4])
##data type of domestic visitors##
df.dtypes
df.columns
## identifying null values in domestic visitors##
df.isnull().sum()
##filling missing values with 0 in domestic visitors#
df['visitors']=df['visitors'].fillna(0)
df.isnull().sum()
## converting visitors column to integer##
df['visitors'] =df['visitors'].replace(' ', )
df['visitors'] =df['visitors'].replace('', )
df['visitors']=df['visitors'].replace('nan',0)
df['visitors']=df['visitors'].astype('int64')
##identifying data type of the domestic visitors data#
df.dtypes
##converting to individual csv file ##
df.to_csv("domestic_visitors.csv")
##filling missing values with 0 in foreign visitors#
df1['visitors']=df1['visitors'].fillna(0)
df1.isnull().sum()
## converting visitors column to integer##
df1['visitors'] =df1['visitors'].replace(' ', )
df1['visitors'] =df1['visitors'].replace('', )
df1['visitors']=df1['visitors'].replace('nan',0)
df1['visitors']=df1['visitors'].astype('int64')
###converting to csv file##
df1.to_csv("foreign_visitors.csv")
df1.dtypes
##identifying number of districts##
sumdistrict=df.groupby(['district']).sum()
print(sumdistrict)
## Top 10 domestic visitors ##
domestic10=sumdistrict.nlargest(10,'visitors')
size=domestic10['visitors'] 
labels=domestic10.index
plt.rcParams['figure.figsize']=(10,10)
##pie plot of top 10 domestic vistiors##
plt.pie(size, labels = labels,shadow  = True, autopct = '%.2f%%')
plt.title('Top 10 domestic visitors ', fontsize = 20)
plt.axis('off')
plt.legend(loc ='upper right')
plt.show() 
###bar plot of top 10 domestic vistors count##
domestic10['visitors'].plot.bar()
plt.title("Top 10 domestic visitors ", fontsize=20)
plt.xlabel('Visitors',fontsize=10)
plt.ylabel('Count',fontsize=10)
plt.show()
##identifying unique districts from dataframe##
district=pd.DataFrame(df['district'].unique())
##caluculating CAGR for each district from 2016-2019##
y=[]
for i in district[0]:
     f=(df[df.district==i])
     x=((((f['visitors'].iloc[-1]/f['visitors'].iloc[0])**(1/3))-1))
     print(x)
     y.append(x)
     print ('Your investment had a CAGR of {:.2%} '.format((x)))
district['CAGR']=pd.DataFrame(y)
##highest CAGR for all districts from 2016-2019##
highestcagr=district.nlargest(10,'CAGR')
##Top CAGR districts bar plot##
plt.bar(highestcagr[0], highestcagr['CAGR'],  width = 0.5)
plt.title("Top CAGR Districts ", fontsize=20)
plt.xlabel('Districts',fontsize=20)
plt.ylabel('CAGR',fontsize=20)
plt.show()
##lowest CAGR for all districts from 2016-2019##
lowestcagr=district.nsmallest(10,'CAGR')
plt.bar(lowestcagr[0], lowestcagr['CAGR'],  width = 0.5)
plt.title("Top CAGR Districts ", fontsize=20)
plt.xlabel('Districts',fontsize=20)
plt.ylabel('CAGR',fontsize=20)
plt.show()
##hyderabad district ##
hyderabad=(df[df['district']=='Hyderabad'])
month=hyderabad.groupby(['month']).sum()
##peak month of hyderabad district##
high=month.nlargest(6,['visitors'])
##pie plot of Peak months hyderabad visitors ##
plt.pie(high['visitors'], labels = high.index,shadow  = True, autopct = '%.2f%%')
plt.title('Peak Season Months Hyderabad Visitors ', fontsize = 20)
plt.axis('off')
plt.legend()
plt.show() 
##pie plot of Low months hyderabad visitors ##
low=month.nsmallest(6,['visitors'])
plt.pie(low['visitors'], labels = low.index,shadow  = True, autopct = '%.2f%%')
plt.title('Low Season Months Hyderabad Visitors ', fontsize = 20)
plt.axis('off')
plt.legend()
plt.show() 
##concatinating all domestic and foreign visitors##
allvisitors=pd.concat([df['district'], df['date'], df['month'], df['year'], df['visitors'],df1['visitors']],axis=1)
allvisitors=allvisitors.groupby(['district']).sum()
domesticsum=df.groupby(['district']).sum()
foreignsum=df1.groupby(['district']).sum()
##ratio of domestic and foreign visitors##
allvisitors['ratio']=domesticsum['visitors']/foreignsum['visitors']

##top domestic to foreign visitors ratio##
top=allvisitors.nlargest(20,'ratio')
plt.bar(top.index, top['ratio'],  width = 0.1)
plt.title("Top domestic to foreign visitors ratio ", fontsize=20)
plt.xlabel('Districts',fontsize=10)
plt.ylabel('Count',fontsize=10)
plt.show()
##Bottom domestic to foreign visitors ratio##
bottom=allvisitors.nsmallest(6,'ratio')
plt.bar(bottom.index, bottom['ratio'],  width = 0.5)
plt.title("Bottom domestic to foreign visitors ratio ", fontsize=20)
plt.xlabel('Districts',fontsize=20)
plt.ylabel('Ratio',fontsize=20)
plt.show()
#population 2019##
population2019=pd.read_excel("C:/Users/user/OneDrive/Desktop/Tourism/C5 Input for participants/population.xlsx")
population2019.columns
population2019=population2019[['District Name', 'Total Population']]
population2019.index = population2019["District Name"]
population2019=population2019.drop(['District Name'],axis=1)
districtdomes=(df[df['year']==2019])
districtforei=(df1[df1['year']==2019])
domes2019=districtdomes.groupby(['district']).sum()
forei2019=districtforei.groupby(['district']).sum()
totalvisitors=(sum(domes2019['visitors'],forei2019['visitors']))
totalvisitors=pd.DataFrame(totalvisitors)
totalvisitors.columns                                       
totalvisitors.index
##population for tourist footfall ratio##
footfall=pd.concat([population2019,totalvisitors],axis=1,join='inner')
footfall.columns
footfall['ratio']=pd.DataFrame(footfall['visitors']/footfall['Total Population'])
#top footfall ratio#
topfootfall=footfall.nlargest(5,'ratio')
plt.bar(topfootfall.index, topfootfall['ratio'],  width = 0.5)
plt.title("Population to Tourist Footfall ratio ", fontsize=20)
plt.xlabel('Districts',fontsize=10)
plt.ylabel('Footfall ratio',fontsize=10)
plt.show()
#bottom footfall ratio##
bottomfootfall=footfall.nsmallest(5,'ratio')
plt.bar(bottomfootfall.index, bottomfootfall['ratio'],  width = 0.5)
plt.title("Population to Tourist Footfall ratio ", fontsize=20)
plt.xlabel('Districts',fontsize=10)
plt.ylabel('Footfall ratio',fontsize=10)
plt.show()
##growth of hyderabad district for each year##
year=hyderabad.groupby(['year']).sum()
year['Growth_Rate'] = year['visitors'].pct_change(periods=1) * 100           
plt.plot(year.index,year['Growth_Rate'])
plt.title("Growth Rate ", fontsize=20)
plt.xlabel('Year',fontsize=20)
plt.ylabel('Growth Rate',fontsize=20)
plt.show()
##

hyderabad.date = pd.to_datetime(hyderabad.date, format='%d-%m-%Y')
plt.ylabel('Hyderabad visitors')
plt.xlabel('Date')
plt.xticks(rotation=45)
plt.plot(hyderabad.date,hyderabad.visitors )

hyderabad =hyderabad.set_index(hyderabad.date )
hyderabad=hyderabad.drop(['date','district', 'month', 'year'],axis=1)
##test and train split##
train = hyderabad[hyderabad.index < pd.to_datetime("2019-01-01", format='%Y-%m-%d')]
test = hyderabad[hyderabad.index > pd.to_datetime("2018-12-01", format='%Y-%m-%d')]
#set datetime as index

plt.plot(train.index,train.visitors, color = "black")
plt.plot(test.index,test.visitors, color = "red")
plt.ylabel('Visitors')
plt.xlabel('Date')
plt.xticks(rotation=45)
plt.title("Train/Test split for Hyderabad district")
plt.show()

import pmdarima as pm
model=pm.auto_arima(train.visitors,start_p=0, d=1, start_q=0, 
                          max_p=5, max_d=5, max_q=5, start_P=0, 
                          D=1, start_Q=0, max_P=5, max_D=5,
                          max_Q=5, m=12, seasonal=True, 
                          error_action='warn',trace = True,
                          supress_warnings=True,stepwise = True,
                          random_state=20,n_fits = 50)
#Summary of the model
model.summary()

prediction = pd.DataFrame(model.predict(n_periods = 12),index=test.index)
prediction.columns = ['predicted_visitors']
prediction

plt.figure(figsize=(8,5))
plt.plot(train.visitors,label="Training")
plt.plot(test.visitors,label="Test")
plt.plot(prediction,label="Predicted")
plt.legend(loc = 'upper right')
plt.show()

from sklearn.metrics import r2_score
test['predicted_visitors'] = prediction
r2_score(test['visitors'], test['predicted_visitors'])

##for future dates##
# Forecast for next 84 days

future=pd.DataFrame(model.predict(n_periods=84))

plt.figure(figsize=(8,5))
plt.plot(train.visitors,label="Training")
plt.plot(test.visitors,label="Test")
plt.plot(future,label="Future")
plt.legend(loc = 'upper right')
plt.show()

##future growth rate 2025 hyderabad#
future['Growth_Rate'] = future[0].pct_change(periods=1) * 100           
plt.plot(future.index,future['Growth_Rate'])
plt.title("Growth Rate ", fontsize=20)
plt.xlabel('Year',fontsize=20)
plt.ylabel('Growth Rate',fontsize=20)
plt.show()


