#911 Calls Project
#Data from kaggle.com
#https://www.kaggle.com/mchirico/montcoalert

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

df = pd.read_csv('911.csv')

df.info()

df.head()

#What are top 5 911 call zip codes?
df['zip'].value_counts().nlargest(5)

#What are top 5 911 call townships(twp)?
df['twp'].value_counts().nlargest(5)

#How many unique title codes are there in 'title' column?
df['title'].nunique()

#Make a new column that returns the reason of the column using lambda
df['REASON']=df['title'].apply(lambda s:s.split(':')[0])

df.head()

#What is the most common reason for a 911 call?
df['REASON'].value_counts().nlargest(1)

#Using seaborn to make a countplot of "REASON" data
sns.countplot(x='REASON',data=df)

type(df['timeStamp'][0])

#Convert timeStamp to date time format

df['timeStamp']=pd.to_datetime(df['timeStamp'])

time = df['timeStamp'].iloc[0]

time.month

df['Month']=df['timeStamp'].apply(lambda t:t.month)

df['Hour']=df['timeStamp'].apply(lambda t:t.hour)

df['Day Of The Week']=df['timeStamp'].apply(lambda t:t.dayofweek)

#Change the days of the week
dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
df['Day Of The Week'].map(dmap)

#Using seaborn to make a count plot of 'day of the week' with hue of 'reason'

sns.countplot(x = df['Day Of The Week'], data = df, hue = df['REASON'])

#Using seaborn to make a count plot of 'Month' with hue of 'reason'

sns.countplot(x = df['Month'], data = df, hue = df['REASON'])

#Months 9,10, 11 are missing. Need to aggregrate
byMonth = df.groupby('Month').count()

byMonth.head()

byMonth['REASON'].plot.line()

#Checking seaborn regression plot
#Reset index to column
sns.lmplot(x='Month',y='REASON',data=byMonth.reset_index())


#Create a new column for dates using the .date method
#Applying this with a lambda function
df['Date'] = df['timeStamp'].apply(lambda t:t.date())

df.groupby('Date').count()['REASON'].plot(figsize=(12,3))


#Separaing out the plots for each 'REASON'
df[df['REASON']=='Fire'] #Selecting the FIRE Data
df[df['REASON']=='Fire'].groupby('Date').count()['REASON'].plot(figsize=(12,3)) #Plotting the data
plt.title('FIRE')

df[df['REASON']=='EMS'] #Selecting the EMS Data
df[df['REASON']=='EMS'].groupby('Date').count()['REASON'].plot(figsize=(12,3)) #Plotting the data
plt.title('EMS')

df[df['REASON']=='Traffic'] #Selecting the Traffic Data
df[df['REASON']=='Traffic'].groupby('Date').count()['REASON'].plot(figsize=(12,3)) #Plotting the data
plt.title('Traffic')

#Preparation for heatmap studies
#Map for better day viewing
df['Day Of The Week']=df['Day Of The Week'].map(dmap)
#Make pivot table
dayHour = df.groupby(by=['Day Of The Week','Hour']).count()['REASON'].unstack()
dayHour.head()

#Generate heat map
sns.heatmap(dayHour,cmap='magma',linecolor='white',linewidths=0.5)#modified

#Cluster map inlcuded
sns.clustermap(dayHour,cmap='magma',linewidths=0.5)

#Preparations of month in column as a heat table
dayMonth = df.groupby(by=['Day Of The Week','Month']).count()['REASON'].unstack()
dayMonth

#Making heat plot of dayMonth
sns.heatmap(dayMonth,cmap='magma',linecolor='white',linewidths=0.5)#modified

#Clustermap of above
sns.clustermap(dayMonth,cmap='magma',linewidths=0.5)
