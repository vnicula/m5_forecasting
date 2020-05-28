import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


INPUT_DIR_PATH = 'm5-forecasting-accuracy/'

def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics: 
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


def read_data():
    sell_prices_df = pd.read_csv(INPUT_DIR_PATH + 'sell_prices.csv')
    sell_prices_df = reduce_mem_usage(sell_prices_df)
    print('Sell prices has {} rows and {} columns'.format(sell_prices_df.shape[0], sell_prices_df.shape[1]))

    calendar_df = pd.read_csv(INPUT_DIR_PATH + 'calendar.csv')
    calendar_df = reduce_mem_usage(calendar_df)
    print('Calendar has {} rows and {} columns'.format(calendar_df.shape[0], calendar_df.shape[1]))

    sales_train_validation_df = pd.read_csv(INPUT_DIR_PATH + 'sales_train_validation.csv')
    print('Sales train validation has {} rows and {} columns'.format(sales_train_validation_df.shape[0], sales_train_validation_df.shape[1]))

    submission_df = pd.read_csv(INPUT_DIR_PATH + 'sample_submission.csv')
    return sell_prices_df, calendar_df, sales_train_validation_df, submission_df


def get_dates_list(calendar_df):
    #Create date index
    date_index = calendar_df['date']
    dates = date_index[0:1913]
    dates_list = [dt.datetime.strptime(date, '%Y-%m-%d').date() for date in dates]
    return dates_list


def get_data_for_store_dept(sales_df, dates_list, store_id, dept_id):

    store_sales_df = sales_df[(sales_df['store_id'] == store_id) & (sales_df['dept_id'] == dept_id)]
    store_sales_df['item_store_id'] = store_sales_df.apply(lambda x: x['item_id']+'_'+x['store_id'], axis=1)
    DF_Sales = store_sales_df.loc[:,'d_1':'d_1913'].T
    DF_Sales.columns = store_sales_df['item_store_id'].values

    #Set Dates as index 
    DF_Sales = pd.DataFrame(DF_Sales).set_index([dates_list])
    DF_Sales.index = pd.to_datetime(DF_Sales.index)

    return DF_Sales


def plot_item(store_sales_df, dates_list, item_index):
    #Select arbitrary index and plot the time series
    y = pd.DataFrame(store_sales_df.iloc[:,item_index])
    TS_selected = y 
    y = pd.DataFrame(y).set_index([dates_list])
    y.index = pd.to_datetime(y.index)
    ax = y.plot(figsize=(20, 10),color='black')
    ax.set_facecolor('lightgrey')
    plt.xticks(fontsize=21 )
    plt.yticks(fontsize=21 )
    plt.legend(fontsize=20)
    plt.title(label = 'Sales Demand Selected Time Series Over Time',fontsize = 23)
    plt.ylabel(ylabel = 'Sales Demand',fontsize = 21)
    plt.xlabel(xlabel = 'Date',fontsize = 21)
    plt.show()