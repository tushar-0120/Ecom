import streamlit as st
st.set_page_config(page_title="My Final Project")
st.title("RFM model")
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    # do something with the file, e.g. save it to a folder
    with open("folder/filename.txt", "wb") as f:
        f.write(uploaded_file.getbuffer())
        st.write("File saved successfully!")

import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

from sklearn import datasets
from sklearn import preprocessing
from sklearn.preprocessing import scale
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples
from sklearn.cluster import AgglomerativeClustering


# Importing the dataset
data=pd.read_csv("E-com_Data.csv")
st.write(data.info())
data=pd.DataFrame(data)
data['Date of purchase'] = pd.to_datetime(data['Date of purchase'], format='%m/%d/%Y')


### DATA CLEANING & WRANGLING ###
## Shape of Data ## isko print krna h
st.title("Data Fetched")
st.write("DATA size : ")
st.write(data.shape) # ye bhi krna h

## Checking for Missing Values ##
data.isnull().sum().sort_values(ascending=False)

## Removing Blank Customer ID from Data Set - 25% data removed
data1 = data.dropna(subset=['CustomerID'])
data1.isnull().sum().sort_values(ascending=False)

## Selecting duplicate rows except first ##
# occurrence based on all columns
duplicate = data1[data1.duplicated()]
print("Duplicate Rows :")
# Print the resultant Dataframe
#duplicate

# Dropping Duplicates from dataset - 8 rows were dropped #
data1 = data1.drop_duplicates(subset=None,keep='first',inplace=False)
data1.info()
data1.head()

## Fixing datatype ##
data1['CustomerID'] = data1['CustomerID'].astype('int64')
data1['CustomerID'] = data1['CustomerID'].astype('str')
data1['InvoiceNo'] = data1['InvoieNo'].astype('str')
data1 = data1.drop(['InvoieNo'],axis=1)
data1.info()
data1.head()

# plot the graph       #1 plot
# Price Column analysis ##
# Plot distribution of Price
st.title("Price Column analysis")
plt.figure(figsize=(12,10))
plt.subplot(2, 1, 1); sns.distplot(data1['Price'])
st.pyplot(plt) ## check
plt.clf()
plt.subplot(2, 2, 2); sns.boxplot(data1['Price'])
st.pyplot(plt)
plt.clf()

data1.Price.describe()
data1.Price[data1['Price'] < 0].count()

# Deleting CustomerID where sum of Price is <= 0 #
z = pd.DataFrame(data1.groupby(['CustomerID'])['Price'].sum()<=0)
z.columns = ['DI']
z.DI.value_counts()

data1 = pd.merge(data1, 
                     z, 
                     on ='CustomerID', 
                     how ='left')

data1 = data1[data1['DI'] == False]
data1 = data1.drop(['DI'], axis=1)
data1.info()

## Date of Purchase Analysis ##
# Extracting Date of Purchase in desired format and setting max date for full data #
data1['date']= pd.to_datetime(data1['Date of purchase'])
print ("First Order Date is",data1['date'].min(), "| Last Order Date is", data1['date'].max())
st.write ("First Order Date is",data1['date'].min(), "| Last Order Date is", data1['date'].max())

# Range of Days in the DataSet #
date_range = data1['date'].max()-data1['date'].min()
days = date_range.days
print(days)
st.write("Range of Days in the DataSet : " , days)

# set Max Date to a variable #
max_date=data1['date'].max()

# extracting Month&Year to a new variable #
data1.insert(loc =2, column='year_month', value=data1['Date of purchase'].map(lambda x: 100*x.year + x.month))

a = pd.DataFrame(data1.year_month.value_counts())
a.reset_index(level=0, inplace=True)

st.title("Count of Transactions per MonthYear")
agraph = sns.barplot(x = 'index', y = 'year_month', data = a)
plt.title("Count of Transactions per MonthYear")
plt.ylabel = "Count"
plt.xlabel = "MonthYear"
plt.xticks(rotation = 90)
st.pyplot(plt)
plt.clf()
#  2
## Drop Columns that are not going to be used ##  @2
data1.info()
data_eda = data1.drop(['Reason of return','Sold as set'], axis=1)
data_eda.info()
data_eda.shape


## EXPLORATORY DATA ANALYSIS ##
st.title("EXPLORATORY DATA ANALYSIS")
# 1. Bar Plot to Show Profile of Data - Count of Transaction, Invoice, CustomerID #
st.write("Bar Plot to Show Profile of Data - Count of Transaction, Invoice, CustomerID")
n = pd.DataFrame(data_eda[['CustomerID', 'Item Code', 'InvoiceNo', 'date']].nunique())
n = n.T
print("No.of.unique values in each column :\n",
      n,)
st.write("No.of.unique values in each column :\n",
      n,)



st.title("Profile of Transaction Data")
graph = sns.barplot(data = n)
plt.title("Profile of Transaction Data")

graph.set(ylabel = "Count")
st.pyplot(plt)
plt.clf()

## 1a. Analysis of Invoice Generation ##
st.title("1a. Analysis of Invoice Generation")
n_orders = pd.DataFrame(data_eda.groupby(['CustomerID'])['InvoiceNo'].nunique())
n_orders.reset_index(level=0, inplace=True)

g = sns.barplot(x = "CustomerID", y= "InvoiceNo", data = n_orders.sort_values('InvoiceNo',ascending=False).head(20))
st.title('Number of Orders per Customer (Top 20)')
g.set_title('Number of Orders per Customer (Top 20)')


plt.xticks(rotation = 90)
sns.color_palette("cubehelix", as_cmap=True)
st.pyplot(plt)
plt.clf()

## 2. Sales & Returns Analysis ##
st.title("2. Sales & Returns Analysis")
# Total Customer and sales #
print (data_eda.CustomerID.nunique(), 'Customers have made $', data_eda.Price.sum(), "in Sales across", data_eda['Item Code'].nunique(), "items" )
st.write (data_eda.CustomerID.nunique(), 'Customers have made $', data_eda.Price.sum(), "in Sales across", data_eda['Item Code'].nunique(), "items" )



# Sales by month 
Total_per_month=data_eda.groupby(['year_month'], as_index=False)['Price'].sum()
Total_per_month

graph1 = sns.barplot(x = 'year_month', y= Total_per_month['Price']/1000000, data = Total_per_month)
graph1.set(ylabel = 'Sales')


plt.title("Sales in $m per month")
plt.xticks(rotation = 90)

sns.color_palette("cubehelix", as_cmap=True)
st.pyplot(plt)
plt.clf()


# Checking for returned products
Return=data_eda[['Item Code','Quantity','Price']].loc[data_eda['Price']<0].sort_values(by='Price', ascending=True)
Return = Return.groupby(['Item Code'], as_index=False).sum().sort_values(by='Price')

Return.Price.sum()/(data_eda.Price.sum()-Return.Price.sum())*100

graph2 = sns.barplot(x='Item Code',y=Return['Price']/1000,data=Return.head(6))
graph2.set(ylabel = 'Price $K')
plt.title('Top 5 Items Returned by Price $K')
st.title('Top 5 Items Returned by Price $K')
sns.color_palette("cubehelix", as_cmap=True)
graph2.scatter([1, 2, 3], [1, 2, 3])
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot()
plt.clf()


## 3. Analysis of Items Sold ##
st.title("3. Analysis of Items Sold")
# Most productive Item codes
Top_products=data_eda[['Item Code','Quantity','Price']].groupby(['Item Code'],as_index=False).sum(['Quantity','Price']).sort_values(by='Quantity',ascending=False)
print (" Top 5 Items Sold by Quantity: \n", Top_products[['Item Code', 'Quantity']].head())
st.write (" Top 5 Items Sold by Quantity: \n", Top_products[['Item Code', 'Quantity']].head())

graph3 = sns.barplot(x='Item Code', y=Top_products['Quantity']/1000, data=Top_products[['Item Code', 'Quantity']].head())
graph3.set(ylabel = 'Quantity 000s')
# @ 8----
sns.color_palette("cubehelix", as_cmap=True)
plt.title('Top Items by Quanity in 000s')
st.title("Top Items by Quanity in 000s")
st.pyplot(plt)
plt.clf()



Top_products_sales=data_eda[['Item Code','Quantity','Price']].groupby(['Item Code'],as_index=False).sum(['Quantity','Price']).sort_values(by='Price',ascending=False)
print (" Top 5 Items by Sales Value: \n", Top_products_sales[['Item Code', 'Price']].head())
st.write (" Top 5 Items by Sales Value: \n", Top_products_sales[['Item Code', 'Price']].head())



graph4 = sns.barplot(x='Item Code', y=Top_products_sales['Price']/1000000, data=Top_products_sales[['Item Code', 'Price']].head())
graph4.set(ylabel = 'Sales in $m')

# 9 -------------
sns.color_palette("cubehelix", as_cmap=True)
plt.title('Top 5 Items by Sales in $m')
st.pyplot(plt)
plt.clf()


Top_products_sales[['Price']].head().sum()/data_eda.Price.sum()*100


## 4. Barplot of Activity by time intervals ##
st.title("4. Barplot of Activity by time intervals")
t = data_eda[['Time','Price','CustomerID','InvoiceNo']]
t.Time = pd.to_datetime(t['Time']).dt.strftime('%H')
ta = t.groupby(['Time'])['CustomerID','InvoiceNo'].nunique()
ta['CustPurch%'] = t.groupby(['Time'])['CustomerID'].nunique()/t.CustomerID.nunique()
ta['Invoice%'] = t.groupby(['Time'])['InvoiceNo'].nunique()/t.InvoiceNo.nunique()
ta.reset_index(level=0, inplace=True)


# @10 ------- 5 krne h yha par
graph5 = sns.barplot(x = 'Time', y = ta['Invoice%']*100, data = ta)
graph5.set(ylim=(0,20))
graph5.yaxis.set_major_formatter(mtick.PercentFormatter())
sns.color_palette("cubehelix", as_cmap=True)
plt.title("Percentage of Invoices per Hour Time Interval")
st.title("Percentage of Invoices per Hour Time Interval")

st.pyplot(plt)
plt.clf()

# 11 6 graph h yha par
graph6 = sns.barplot(x = 'Time', y = ta['CustPurch%']*100, data = ta)
graph6.set(ylim=(0,45))
graph6.yaxis.set_major_formatter(mtick.PercentFormatter())
plt.title("Percentage of Customers per Hour Time Interval")
st.title("Percentage of Customers per Hour Time Interval")
st.pyplot(plt)
plt.clf()

## 5. Barplot of Shipping Location
st.title("Barplots of Shipping Location")
sl = data_eda[['Item Code','Shipping Location','Cancelled_status']].groupby(['Shipping Location'],as_index=False).count().sort_values(by='Item Code',ascending=False)
sl['ShipLoc%'] = sl['Item Code']/data_eda['Item Code'].count()
sl['CancShipLoc%'] = sl['Cancelled_status']/data_eda['Cancelled_status'].count()
sl['CancShipPLoc%'] = sl['Cancelled_status']/sl['Item Code']


# @ 13 ---------
graph7 = sns.barplot(x = 'Shipping Location', y = sl['ShipLoc%']*100, data = sl)
graph7.yaxis.set_major_formatter(mtick.PercentFormatter())
sns.color_palette("cubehelix", as_cmap=True)
plt.xticks(rotation = 90)
plt.title("Percentage of Total Items per Shipping Location")
st.title("Percentage of Total Items per Shipping Location")
st.pyplot(plt)
plt.clf()


graph8 = sns.barplot(x = 'Shipping Location', y = sl['CancShipPLoc%']*100, data = sl)
graph8.yaxis.set_major_formatter(mtick.PercentFormatter())
sns.color_palette("cubehelix", as_cmap=True)
plt.xticks(rotation = 90)
plt.title("Percentage of Cancelled Items Per Total Items in Location")
st.title("Percentage of Cancelled Items Per Total Items in Location")
st.pyplot(plt)
plt.clf()


data_eda['Shipping Location'].nunique()
data_eda.groupby(['Cancelled_status'])['Shipping Location'].nunique()

data_eda.to_csv('data_eda.csv')


### RFM & MODEL DATA ###
st.title("RFM & MODEL DATA")

## Making the column "CustomerID" as the index and dropping unwanted columns ##
data_eda.info()
data_eda.shape
data_rfm = data_eda.drop(['year_month','Item Code','Quantity','Time','price per Unit','Shipping Location','Cancelled_status'], axis=1)
data_rfm.info()


data_rfm=data_rfm.set_index('CustomerID')

## RFM & Model Data - Aggregating Data by Customer ID ##
data_rfm=data_rfm.groupby(['CustomerID']).agg({'InvoiceNo':'nunique', 
                                                         'Price':'sum',
                                                         'date':['min','max']})

data_rfm.columns=['count','price','min_date','max_date']
data_rfm.info()

## Feature Engoineering - Calculating RFM values ##
data_rfm['recency'] = max_date-data_rfm['max_date']
data_rfm['recency'] = data_rfm['recency'].dt.days

data_rfm['frequency'] = data_rfm['count']

data_rfm['monetary_value'] = data_rfm['price']

data_rfm=data_rfm.drop(['count','price','min_date','max_date'],axis=1)

data_rfm.info()
data_rfm.shape
rfm_stats = pd.DataFrame(data_rfm.describe())

skew = pd.DataFrame(data_rfm.skew())


plt.figure(figsize=(12,10))
# Plot distribution of Recency
plt.subplot(3, 1, 1); sns.distplot(data_rfm['recency'])

# Plot distribution of Frequency
plt.subplot(3, 1, 2); sns.distplot(data_rfm['frequency'])
# Plot distribution of Monetary Value
plt.subplot(3, 1, 3); sns.distplot(data_rfm['monetary_value'])
st.pyplot(plt)
plt.clf()


from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
from IPython import get_ipython

# axes instance
#get_ipython().run_line_magic('matplotlib inline', 'qt')
fig = plt.figure(figsize=(6,6))
ax = Axes3D(fig, auto_add_to_figure=False)
fig.add_axes(ax)
# get colormap from seaborn
cmap = ListedColormap(sns.color_palette("husl", 256).as_hex())
# yha bahut sare h 6 line
# plot
sc = ax.scatter(data_rfm.recency, data_rfm.frequency, data_rfm.monetary_value, s=40, c=data_rfm.recency, marker='o', cmap=cmap, alpha=1)
ax.set_xlabel('Recency')
ax.set_ylabel('Frequency')
ax.set_zlabel('Monetary')
# legend 
plt.legend(*sc.legend_elements(), bbox_to_anchor=(1.05, 1), loc=2)
# save
plt.savefig("scatter_hue", bbox_inches='tight')
#get_ipython().run_line_magic('matplotlib', 'inline')
st.pyplot(plt)
plt.clf()

## Statistical RFM Ranking, RFM Grouping, RFM Scoring & Segmentation ##
# set the bins [<min, x, x, x, max]
r_bins = [-1, 17, 50, 140, 380]
f_bins = [0, 1, 3, 5, 243]
m_bins = [0, 38585, 84965, 206819, 35536194]
data_rfm['r_score'] = pd.cut(data_rfm['recency'], r_bins, labels = [4, 3, 2, 1])
data_rfm['f_score'] = pd.cut(data_rfm['frequency'], f_bins, labels = [1, 2, 3, 4])
data_rfm['m_score'] = pd.cut(data_rfm['monetary_value'], m_bins, labels = [1, 2, 3, 4])

data_rfm['r_score'] = data_rfm['r_score'].astype('int64')
data_rfm['f_score'] = data_rfm['f_score'].astype('int64')
data_rfm['m_score'] = data_rfm['m_score'].astype('int64')

data_rfm['rfm_group'] = data_rfm['r_score'].astype('str').str.cat(data_rfm['f_score'].astype('str')).str.cat(data_rfm['m_score'].astype('str'))
data_rfm['rfm_score'] = data_rfm['r_score'] + data_rfm['f_score'] + data_rfm['m_score']
data_rfm['cust_group'] = data_rfm['rfm_score']
data_rfm['cust_group'] = np.where((data_rfm.rfm_score <=5),'Passive', data_rfm.cust_group)
data_rfm['cust_group'] = np.where((data_rfm.rfm_score > 5) & (data_rfm.rfm_score <= 9),'Critical', data_rfm.cust_group)
data_rfm['cust_group'] = np.where((data_rfm.rfm_score > 9), 'Loyal', data_rfm.cust_group)

# Plotting the RFM Grpups #
st.title("Plotting the RFM Grpups")
rfm_groupcount = pd.DataFrame(data_rfm['rfm_group'].value_counts())
rfm_groupcount.reset_index(level=0, inplace=True)
rfm_groupcount.columns=['rfm_group','count']
rfm_groupcount



graph9 = sns.barplot(x = 'rfm_group', y = 'count', data = rfm_groupcount.head(32))
plt.xticks(rotation = 'vertical')
sns.color_palette("cubehelix", as_cmap=True)
graph9.set_title('Count of Cutsomers per RFM Group (Top32)')
st.title('Count of Cutsomers per RFM Group (Top32) ye htana h')
st.pyplot(plt)
plt.clf()

st.title("Plotting the RFM Score")
# Plotting the RFM Score # 
rfmmeansbyscore = data_rfm[['recency','frequency','monetary_value']].groupby(data_rfm['rfm_score']).mean()

rfm_scorecount = pd.DataFrame(data_rfm['rfm_score'].value_counts())
rfm_scorecount.reset_index(level=0, inplace=True)
rfm_scorecount.columns=['rfm_score','count']
rfm_scorecount


graph10 = sns.barplot(x = 'rfm_score', y = 'count', data = rfm_scorecount)
graph10.set_title("Count of RFM Score")
sns.color_palette("cubehelix", as_cmap=True)
st.pyplot()


# Plotting the Customer Groups basis RFM Score #
st.title("Plotting the Customer Groups basis RFM Score")
rfm_scoregroupcount = data_rfm.groupby(['cust_group']).size().reset_index(name='count') 
rfm_scoregroupcount

graph11 = sns.barplot(x = 'cust_group', y = 'count', data = rfm_scoregroupcount)
graph11.set_title("Count of RFM Customer Group")
sns.color_palette("cubehelix", as_cmap=True)
colors=['yellow', 'green', 'orange']
st.pyplot()
plt.clf()

# # change
plt.pie(rfm_scoregroupcount['count'],labels=rfm_scoregroupcount.cust_group, colors=colors, startangle=90, autopct='%1.1f%%')
st.pyplot()
plt.clf()



# 3D Plot based on 9 RFM Scores #
#get_ipython().run_line_magic('matplotlib', 'qt')
fig = plt.figure(figsize=(6,6))
ax = Axes3D(fig, auto_add_to_figure=False)
fig.add_axes(ax)
# get colormap from seaborn
cmap = ListedColormap(sns.color_palette("husl", 256).as_hex())
# plot
sc = ax.scatter(data_rfm.recency, data_rfm.frequency, data_rfm.monetary_value, s=40, c=data_rfm.rfm_score, marker='o', cmap=cmap, alpha=1)
ax.set_xlabel('Recency')
ax.set_ylabel('Frequency')
ax.set_zlabel('Monetary')
# legend
plt.legend(*sc.legend_elements(), bbox_to_anchor=(1.05, 1), loc=2)
#get_ipython().run_line_magic('matplotlib', 'inline')
st.pyplot(plt)
# save
plt.savefig("scatter_hue", bbox_inches='tight')
st.pyplot(plt)
plt.clf()

data_rfm.info()
data_rfm.to_csv('data_rfm.csv')


### CLUSTERING MODEL ###
st.title("CLUSTERING MODEL")
data_rfm1 = data_rfm.copy()
data_rfm1.info()
data_rfm1 = data_rfm1.drop(['r_score','f_score','m_score','rfm_group','rfm_score','cust_group'], axis=1)


## Fix Skewness - Updating outliers ##
plt.boxplot(data_rfm1.frequency)
st.title("Fix Skewness - Updating outliers")
st.pyplot(plt)
plt.clf()

fQ1 = np.quantile(data_rfm1.frequency,.25)
fQ3 = np.quantile(data_rfm1.frequency,.75)
fIQR = fQ3 - fQ1
fUB = fQ3 + 1.5*fIQR
fLB = fQ1 - 1.5*fIQR
fUB

len(data_rfm1[data_rfm1.frequency>fUB])
382/4319
data_rfm1['frequency'] = np.where(data_rfm1['frequency'] > fUB, fUB,data_rfm1['frequency'])


plt.boxplot(data_rfm1.monetary_value)
st.pyplot(plt)
plt.clf()


mQ1 = np.quantile(data_rfm1.monetary_value,.25)
mQ3 = np.quantile(data_rfm1.monetary_value,.75)
mIQR = mQ3 - mQ1
mUB = mQ3 + 1.5*mIQR
mLB = mQ1 - 1.5*mIQR
mUB

len(data_rfm1[data_rfm1.monetary_value>mUB])
415/4319
data_rfm1['monetary_value'] = np.where(data_rfm1['monetary_value'] > mUB, mUB,data_rfm1['monetary_value'])


# 3D DataPlot post Outlier correction ##
st.title("3D DataPlot post Outlier correction")
fig = plt.figure(figsize=(6,6))
ax = Axes3D(fig, auto_add_to_figure=False)
fig.add_axes(ax)
# get colormap from seaborn
cmap = ListedColormap(sns.color_palette("husl", 256).as_hex())
# plot
sc = ax.scatter(data_rfm1.recency, data_rfm1.frequency, data_rfm1.monetary_value, s=40, c=data_rfm.recency, marker='o', cmap=cmap, alpha=1)
ax.set_xlabel('Recency')
ax.set_ylabel('Frequency')
ax.set_zlabel('Monetary')
# legend  
plt.legend(*sc.legend_elements(), bbox_to_anchor=(1.05, 1), loc=2)
st.pyplot(plt)
plt.clf()
data_rfm1.info()
rfm1po_stats = pd.DataFrame(data_rfm1.describe())
skew1po = pd.DataFrame(data_rfm1.skew())

### Testing & Correcting Skewness ###
## Testing for skewness of the data ##
plt.figure(figsize=(12,10))
# Plot distribution of Recency
plt.subplot(3, 1, 1); sns.distplot(data_rfm1['recency'])
# Plot distribution of Frequency
plt.subplot(3, 1, 2); sns.distplot(data_rfm1['frequency'])
# Plot distribution of Monetary Value
plt.subplot(3, 1, 3); sns.distplot(data_rfm1['monetary_value'])

st.pyplot(plt)
plt.clf()
st.title("Performing Log Trasnformation to correct skewness")
## Performing Log Trasnformation to correct skewness ##
data_rfm1['log_recency'] = np.log(data_rfm1['recency']+1)
data_rfm1['log_frequency'] = np.log(data_rfm1['frequency']+1)
data_rfm1['log_monetary_value'] = np.log(data_rfm1['monetary_value']+1)

## Re-Testing for skewness of the data ##
plt.figure(figsize=(12,10))
# Plot distribution of Recency
plt.subplot(3, 1, 1); sns.distplot(data_rfm1['log_recency'])
# Plot distribution of Frequency
plt.subplot(3, 1, 2); sns.distplot(data_rfm1['log_frequency'])
# Plot distribution of Monetary Value
plt.subplot(3, 1, 3); sns.distplot(data_rfm1['log_monetary_value'])
st.pyplot(plt)
plt.clf()

data_rfm1.recency = data_rfm1.log_recency
data_rfm1.frequency = data_rfm1.log_frequency
data_rfm1.monetary_value = data_rfm1.log_monetary_value
data_rfm1 = data_rfm1.drop(['log_recency','log_frequency','log_monetary_value'],axis=1)

data_rfm1.info()

skew1plog = pd.DataFrame(data_rfm1.skew())

from sklearn.preprocessing import StandardScaler
data_scaled = StandardScaler().fit_transform(data_rfm1)

## Applying the Elbow Method ##
plt.figure(figsize = (10, 8))
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init = 'random', random_state = 42)
    kmeans.fit(data_scaled)
    wcss.append(kmeans.inertia_)
    
#wcss # 10 values will appear    

plt.plot(range(1,11),wcss, 'gx-')
plt.title('WCSS of Clusters')
st.title("WCSS of Clusters")
st.pyplot(plt)
plt.clf()


## Running KMeans to our desired number of optimal clusters (k = 3)
# making clusters in the backend not fitted to dataset #
kmeans = KMeans(n_clusters = 3, random_state = 42)

# fitting the cluster to the dataset #
clusters = kmeans.fit_predict(data_scaled)

# cluster allocation #
clusters
Final_Clusters = clusters + 1
cluster = list(Final_Clusters)

## Overall Silhouette score ##
st.title("Overall Silhouette score")
print(f'sil score(n=3): {silhouette_score(data_scaled,cluster)}')
st.write(f'sil score(n=3): {silhouette_score(data_scaled,cluster)}')

# Silhouette score of each data point #
sample_silhouette_values = silhouette_samples(data_scaled, cluster)
sample_silhouette_values = pd.DataFrame(sample_silhouette_values)
sample_silhouette_values.to_csv('sil_values3.csv')



# CLUSTER MEMBERSHIP #
st.title("Cluster Membership")
data_final = data_rfm.copy()

data_rfm1['Cluster'] = cluster
data_final['Cluster'] = cluster

sample_silhouette_values = silhouette_samples(data_scaled, cluster)
sil_score = list(sample_silhouette_values)
data_rfm1['sil_score'] = sil_score
data_final['sil_score'] = sil_score

data_final.info()

## Cluster wise mean r/f/m/sil values ##
cm = pd.DataFrame(data_final[['recency','frequency','monetary_value','sil_score']].groupby(data_final['Cluster']).mean())


## Pattern of sil scores across clusters ##
sns.boxplot(x = 'Cluster', y = 'sil_score', data = data_final)
plt.title('Boxplot of Silhouette Scores')
st.title("Boxplot of Silhouette Scores")
st.pyplot(plt)
plt.clf()

### Analysis on Model Data ###
## Profiling of Clusters ##
data_final['mcust_group'] = data_final['Cluster']
data_final['mcust_group'] = np.where((data_final.Cluster == 1), 'Passive', data_final.mcust_group)
data_final['mcust_group'] = np.where((data_final.Cluster == 2), 'Loyal', data_final.mcust_group)
data_final['mcust_group'] = np.where((data_final.Cluster == 3), 'Critical', data_final.mcust_group)

data_final.info()
data_final.to_csv('data_rfmmodelfinal.csv')


mcust_groupcount = data_final.groupby(['mcust_group']).size().reset_index(name='count') 
mcust_groupcount


graph12 = sns.barplot(x = 'mcust_group', y = 'count', data = mcust_groupcount)
sns.color_palette("cubehelix", as_cmap=True)
graph12.set_title("Count of Model Customer Group")
st.title("Count of Model Customer Group")
st.pyplot(plt)
plt.clf()
colors=['yellow', 'green', 'orange']
plt.pie(mcust_groupcount['count'],labels=mcust_groupcount.mcust_group, colors=colors, startangle=90, autopct='%1.1f%%')

st.pyplot(plt)
plt.clf()
pd.crosstab(data_final['rfm_score'],data_final['mcust_group'])
cross_table=pd.crosstab(data_final['rfm_score'],data_final['mcust_group'])/4319*100
cross_table


fig, ax = plt.subplots(figsize=(8, 5))
sns.heatmap(cross_table/100,
                annot=True,
                fmt='.2%',
                #cmap='rocket_r',
                #linewidths=.5,
                ax=ax)
ax.set_title("CrossTab of Modeled Customer Groups with Statitical RFM Score")
plt.show()
st.pyplot(plt)
plt.clf()


## 3D plotting of RFM based on Transformed Data using final clusters ##
st.title("3D plotting of RFM based on Transformed Data using final clusters")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from IPython import get_ipython

#get_ipython().run_line_magic('matplotlib', 'qt')
r=data_rfm1.recency
f=data_rfm1.frequency
m=data_rfm1.monetary_value

fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection='3d')
fig = ax.scatter(r,f,m, c=data_final.Cluster, cmap = "spring")
ax.set_title("Cluster-wise Recency, Frequency, MonetaryValue distribution")
ax.set_xlabel("Recency")
ax.set_ylabel("Frequency")
ax.set_zlabel("MonetaryValue")
st.title("Cluster-wise Recency, Frequency, MonetaryValue distribution")
st.pyplot()
#get_ipython().run_line_magic('matplotlib', 'inline')