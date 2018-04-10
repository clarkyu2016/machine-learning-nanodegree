import pandas as pd
import numpy as np
from datetime import datetime, timedelta

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
store = pd.read_csv('store.csv')

#1、去除销量中的极值,由于无法估计极值对结果的影响，所以拟合模型的时候可以进行两次，去除极值和未去除极值
#再试一下标准差标准误差
def rm_outliers(df): 
    q1 = np.percentile(df['Sales'], 25, axis=0)
    q3 = np.percentile(df['Sales'], 75, axis=0)
    k = 2.5
    iqr = q3 - q1
    df = df[df['Sales'] > q1 - k*iqr]
    df = df[df['Sales'] < q3 + k*iqr]
    return df

def rm_outliers_std(df): 
    std = df['Sales'].std()
    mean = df['Sales'].mean()
    k = 3
    df = df[df['Sales'] >  mean - k*std]
    df = df[df['Sales'] < mean + k*std]
    return df

#2、对时间的拆分
def data_processing(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df['DayOfYear'] = df['Date'].apply(lambda x: x.dayofyear)
    df['WeekOfYear'] = df['Date'].apply(lambda x: x.week)
    df['Month'] = df['Date'].apply(lambda x: x.month)
    df['DayOfMonth'] = df['Date'].apply(lambda x: x.day)
    df['Year'] = df['Date'].apply(lambda x: x.year)
    return df

#4、为每个日期添加过去一个季度，过去半年，过去一年，过去两年的这家店的平均日销售量
def store_sales_each_day(sale):
    
    def add_mean(store,sale,current_date,past_time):
        past_date = current_date - timedelta(days=past_time)
        mean_sale = sale[(sale['Date'] < current_date) & (sale['Date'] > past_date) & (sale['Store'] == store)]['Sales'].mean()
        return mean_sale
    
    sale['past_quater_mean_sale'] = sale.apply(lambda row: add_mean(row['Store'], sale, row['Date'], 90), axis=1)
    sale['past_year_mean_sale'] = sale.apply(lambda row: add_mean(row['Store'], sale, row['Date'], 183), axis=1)
    
    return sale

#测试集调整
def store_sales_each_day_for_test(sale,test):
    
    def add_mean(store,sale,current_date,past_time):
        past_date = current_date - timedelta(days=past_time)
        mean_sale = sale[(sale['Date'] < current_date) & (sale['Date'] > past_date) & (sale['Store'] == store)]['Sales'].mean()
        return mean_sale
    
    test['past_quater_mean_sale'] = test.apply(lambda row: add_mean(row['Store'], sale, row['Date'], 90), axis=1)
    test['past_year_mean_sale'] = test.apply(lambda row: add_mean(row['Store'], sale, row['Date'], 183), axis=1)

    return test

#3、为每家店添加销售量客流量相关的均值,执行顺序在对时间进行拆分后
def add_mean_for_store(sales,store_df=store) :
    mean_sales_promo = []
    mean_sales_no_promo = []
    mean_sales = []
    mean_sales_2013 = []
    mean_sales_2014 = []
    mean_sales_2015 = []
    mean_store_sales_1month = []
    mean_store_sales_2months = []
    mean_store_sales_3months = []
    mean_store_sales_6months = []
    
    
    mean_customers_promo = []
    mean_customers_no_promo = []
    mean_customers = []
    mean_customers_2013 = []
    mean_customers_2014 = []
    mean_customers_2015 = []
    mean_customers_1month = [] 
    mean_customers_2months = [] 
    mean_customers_3months = []
    mean_customers_6months = []
    
    
    for store in store_df['Store']:
        sale = sales[sales['Store']==store]
        
        # mean of sales
        mean_sales.append(sale['Sales'].mean())
        mean_sales_promo.append(sale[sale['Promo'] == 1]['Sales'].mean())
        mean_sales_no_promo.append(sale[sale['Promo'] == 0]['Sales'].mean())
        mean_sales_2013.append(sale[sale['Year'] == 2013]['Sales'].mean())
        mean_sales_2014.append(sale[sale['Year'] == 2014]['Sales'].mean())
        mean_sales_2015.append(sale[sale['Year'] == 2015]['Sales'].mean())
        mean_store_sales_1month.append(sale[(sale['Month'] == 7) & (sale['Year'] == 2015)]['Sales'].mean())                   
        mean_store_sales_2months.append(sale[(sale['Month'] <= 7) ^(sale['Month'] >= 6) & (sale['Year'] == 2015)]['Sales'].mean())
        mean_store_sales_3months.append(sale[(sale['Month'] <= 7) ^(sale['Month'] >= 5) & (sale['Year'] == 2015)]['Sales'].mean())        
        mean_store_sales_6months.append(sale[(sale['Month'] <= 7) ^(sale['Month'] >= 2) & (sale['Year'] == 2015)]['Sales'].mean())
        
        # mean of customers
        mean_customers.append(sale['Customers'].mean())
        mean_customers_promo.append(sale[sale['Promo'] == 1]['Customers'].mean())
        mean_customers_no_promo.append(sale[sale['Promo'] == 0]['Customers'].mean())
        mean_customers_2013.append(sale[sale['Year'] == 2013]['Customers'].mean())
        mean_customers_2014.append(sale[sale['Year'] == 2014]['Customers'].mean())
        mean_customers_2015.append(sale[sale['Year'] == 2015]['Customers'].mean())
        mean_customers_1month.append(sale[(sale['Month'] == 7) & (sale['Year'] == 2015)]['Customers'].mean())                   
        mean_customers_2months.append(sale[(sale['Month'] <= 7) ^(sale['Month'] >= 6) & (sale['Year'] == 2015)]['Customers'].mean())
        mean_customers_3months.append(sale[(sale['Month'] <= 7) ^(sale['Month'] >= 5) & (sale['Year'] == 2015)]['Customers'].mean())        
        mean_customers_6months.append(sale[(sale['Month'] <= 7) ^(sale['Month'] >= 2) & (sale['Year'] == 2015)]['Customers'].mean())
        
    store_df['mean_sales'] = mean_sales
    store_df['mean_sales_promo'] = mean_sales_promo
    store_df['mean_sales_no_promo'] = mean_sales_no_promo
    store_df['mean_sales_2013'] = mean_sales_2013
    store_df['mean_sales_2014'] = mean_sales_2014
    store_df['mean_sales_2015'] = mean_sales_2015
    store_df['mean_store_sales_1month'] = mean_store_sales_1month
    store_df['mean_store_sales_2months'] = mean_store_sales_2months
    store_df['mean_store_sales_3months'] = mean_store_sales_3months
    store_df['mean_store_sales_6months'] = mean_store_sales_6months
    
    store_df['mean_customers'] = mean_customers
    store_df['mean_customers_promo'] = mean_customers_promo
    store_df['mean_customers_no_promo'] = mean_customers_no_promo
    store_df['mean_customers_2013'] = mean_customers_2013
    store_df['mean_customers_2014'] = mean_customers_2014
    store_df['mean_customers_2015'] = mean_customers_2015
    store_df['mean_customers_1month'] = mean_customers_1month
    store_df['mean_customers_2months'] = mean_customers_2months
    store_df['mean_customers_3months'] = mean_customers_3months
    store_df['mean_customers_6months'] = mean_customers_6months
    
    return store_df

def drop_stores(data_test, data):
	stores = data_test['Store'].unique()         
	return data[data_test['Store'].isin(stores)]

#合并销售和商店 
def merge_sale(sale_data, store_data):
    train = sale_data.join(store_data, on='Store', rsuffix='_')
    train = train.drop('Store_',axis=1)
    return train

#添加其他特征
def extra_features(data):
    data['CompetitionOpen'] = 12 * (data['Year'] - data['CompetitionOpenSinceYear']) + (data['Month'] - data['CompetitionOpenSinceMonth'])
    data['PromoOpen'] = 12 * (data['Year'] - data['Promo2SinceYear'])+ (data['WeekOfYear'] - data['Promo2SinceWeek']) / 4.0
    data['PromoOpen'] = data['PromoOpen'].apply(lambda x: x if x > 0 else 0)
    data.loc[data.Promo2SinceYear == 0, 'PromoOpen'] = 0
    data = data.drop(['CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'Promo2SinceWeek', 'Promo2SinceYear'], axis=1) #删除特征
    
    mappings = {'0': 0, 'a': 1, 'b': 2, 'c': 3, 'd': 4,'Jan,Apr,Jul,Oct':1,'Feb,May,Aug,Nov':2,'Mar,Jun,Sept,Dec':3}  
    data['StoreType'].replace(mappings, inplace=True)  
    data['Assortment'].replace(mappings, inplace=True)  
    data['StateHoliday'].replace(mappings, inplace=True)
    data['PromoInterval'].replace(mappings, inplace=True)
    data = data.drop(['Date'],axis= 1)
    return data

#去除极大极小值
print('Moving outliers')
train = rm_outliers(train)

#转换年月日
print('Convert Time')
train = data_processing(train)
test = data_processing(test)

#给商店计算平均销售量
store = add_mean_for_store(train)
store = drop_stores(test, store)

print('add additional past_quater_mean_sale and past_year_mean_sale')
#添加额外特征'past_quater_mean_sale'和'past_year_mean_sale'，这段代码运行时间可能会过长
train = store_sales_each_day(train)
test = store_sales_each_day_for_test(train,test)

print('Merging')
#合并
train = merge_sale(train, store)
test = merge_sale(test, store)


#额外的特征
train = extra_features(train)
test = extra_features(test)

holidayofyear = sorted(train[train['StateHoliday'].isin([1,2,3,4])]['DayOfYear'].unique())
def day2holiday(df,holidayofyear):
    for holiday in holidayofyear:
        df['DaysToHoliday' + str(holiday)] = holiday - df['DayOfYear']
    return df

#计算距离节日的日子数
train = day2holiday(train,holidayofyear)
test =  day2holiday(test,holidayofyear)

print('Final output')
#生成最终输入
train.to_csv('train_withextra.csv',index=False)
test.to_csv('test_withextra.csv',index=False)
