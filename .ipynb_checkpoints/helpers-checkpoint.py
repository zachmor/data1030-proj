import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt, floor
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, MinMaxScaler

class View:
    
    def quiver_recent_metadata(self):

        # gathering quiver_recent file paths from data directory
        quiver_recent_files = glob.glob(os.path.join(path, "data/quiver_recent/*.csv"))

        # loading most of them into a dictionary with readable key names, e.g. "congress_trading" 
        recent_datasets = {f[63:-4] : pd.read_csv(f, index_col=0) 
            for f in quiver_recent_files }
            #if f[63:-4] not in ['congress_trading', 'flights', 'political_beta']}

        # gathering metadata for each dataset
        metadata  = {
            key : { 
                    'name' : key,
                    "shape" : recent_datasets[key].shape,
                    "columns" : recent_datasets[key].columns,
                    "n_columns" : len(recent_datasets[key].columns),
                    "n_rows" : len(recent_datasets[key]),
                    "tickers" : recent_datasets[key]["Ticker"].unique(),
                    "ticker_counts" : recent_datasets[key]["Ticker"].value_counts().head()
                }
            for key in recent_datasets }

        df = pd.DataFrame(metadata).transpose()

        return df

    # loads in the insider trading dataset
    def msft_insider_trading(self):

        df = pd.read_csv('data/quiver/insiders.csv', parse_dates=['Date'])

        return df
    
    
    # return a class for each point, 
    # 0 = undervalued valued if within window period for at least confirmation time if goes above threshold
    # 1 = overvalued if within window period for at least confirmation time if goes below threshold
    # 2 = efficient if it's not undervalued or overvalued
    # 3 = volatile if it's undervalued and overvalued

class Visualize:
    
    # How many datasets and columns per dataset from quiver_recent_metadata view
    def quiver_diversity(self, df):
        
        fig, ax = plt.subplots(figsize=[5,5])

        plt.bar(x=df['name'], height=df['n_columns'], color=list('rgbkymc'))
        plt.xticks(rotation=45, ha='right')
        plt.yticks(fontsize=12)
        plt.ylabel('Number of Columns', fontsize=14)
        plt.xlabel('Dataset Name', fontsize=14)
        plt.title('Quiver Live: Datasets and Columns', fontsize=16)
        plt.tight_layout()

        plt.show()

        all_but_wsb = df[df['name'] != 'wallstreetbets']

        print('rows', all_but_wsb['n_rows'].sum())
        print('cols', all_but_wsb['n_columns'].sum())
        print('cols', len(set(all_but_wsb['n_columns'].sum())))
  
    # Demonstrates downsampling with the msft_insider_trading view 
    def grouping_sacrifices(self, df):

        acquired = df[df['AcquiredDisposedCode'] == 'A']
        disposed = df[df['AcquiredDisposedCode'] == 'D']
        
        df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
        
        df2 = df.groupby(['Date', 'AcquiredDisposedCode'])['Shares'].sum().unstack('AcquiredDisposedCode')
            
        print('length before aggregating records', len(df))
        print('length after aggregating records',len(df2))
        
        df2.plot(kind='bar', stacked=True)
        plt.semilogy()
        plt.title('MSFT Insider Trading', fontsize=16)
        plt.ylabel('Shares [#]', fontsize=14)
        plt.xlabel('Time', fontsize=14)
        plt.tight_layout()
        plt.xticks(np.arange(0, len(df2), 2), rotation=60, ha='right')
        plt.show()

        print(df2.head())
        
    # A traditional candleplot from data/alpha 
    def msft_candlesticks(self, slice_start: int=-1, slice_end: int=-1):
        
        if (slice_start > 0) and (slice_end > 0):
            alpha = pd.read_csv("data/alpha/MSFT.csv",
                            index_col='time', parse_dates=['time'])[slice_start:slice_end]
        
        elif (slice_start > 0) and (slice_end < 0): 
            alpha = pd.read_csv("data/alpha/MSFT.csv",
                            index_col='time', parse_dates=['time'])[slice_start:]
            
        elif (slice_start < 0) and (slice_end > 0): 
            alpha = pd.read_csv("data/alpha/MSFT.csv",
                            index_col='time', parse_dates=['time'])[:slice_end]
            
        else: # most recent week
            alpha = pd.read_csv("data/alpha/MSFT.csv",
                            index_col='time', parse_dates=['time'])[-200:]

        plt.figure(figsize=[5,5])

        width = sqrt(sqrt(len(alpha)))*.004
        width2 = width/6.2

        up = alpha[alpha['close'] >= alpha['open']]
        down = alpha[alpha['close'] < alpha['open']]

        up_color = 'green'
        down_color = 'red'

        plt.bar(up.index, up.close - up.open, width, bottom = up.open, color=up_color)
        plt.bar(up.index, up.high - up.close, width2, bottom = up.close, color=up_color)
        plt.bar(up.index, up.open - up.low, width2, bottom = up.low, color=up_color)

        plt.bar(down.index, down.open - down.close, width, bottom = down.close, color=down_color)
        plt.bar(down.index, down.high - down.open, width2, bottom = down.open, color=down_color)
        plt.bar(down.index, down.close - down.low, width2, bottom = down.low, color=down_color)

        plt.title('MSFT Stock Price', fontsize=16)
        plt.xlabel('Time [Month-Day Hour]', fontsize=14)
        plt.ylabel('Price [$]', fontsize=14)

        plt.xticks(fontsize=12,rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

        return

    def msft_inefficiencies():

        for i in range(15):
            try:
                assert('inefficiency'+str(i) in df.columns)
            except AssertionError:
                print('this visualization requires 15 inefficiency columns')

        fig, ax = plt.subplots(figsize=[24,5])

        width = sqrt(sqrt(len(df)))*.004
        width2 = width/6.2

        up = df[df['close'] >= df['open']]
        down = df[df['close'] < df['open']]

        up_color = 'green'
        down_color = 'red'

        plt.bar(up.index, up.close - up.open, width, bottom = up.open, color=up_color)
        plt.bar(up.index, up.high - up.close, width2, bottom = up.close, color=up_color)
        plt.bar(up.index, up.open - up.low, width2, bottom = up.low, color=up_color)

        plt.bar(down.index, down.open - down.close, width, bottom = down.close, color=down_color)
        plt.bar(down.index, down.high - down.open, width2, bottom = down.open, color=down_color)
        plt.bar(down.index, down.close - down.low, width2, bottom = down.low, color=down_color)

        ax2 = ax.twinx()

        for i in range(15):

            underpriced = df[df['inefficiency' + str(i)] == 0]
            overpriced = df[df['inefficiency' + str(i)] == 1]
            efficient = df[df['inefficiency' + str(i)] == 2]
            volatile = df[df['inefficiency' + str(i)] == 3]

            ax2.bar(underpriced.index, height=.95, width=width, color='green',
                    bottom=0.5 + i + floor(i/5), alpha=.5, label= 'underpriced')
            ax2.bar(overpriced.index, height=.95, width=width, color='red',
                    bottom=0.5 + i + floor(i/5), alpha=.5, label= 'overpriced')
            ax2.bar(efficient.index,height=.95, width=width, color='blue',
                    bottom=0.5 + i + floor(i/5),alpha=.5, label= 'efficient')
            ax2.bar(volatile.index, height=.95, width=width, color='yellow',
                    bottom=0.5 + i + floor(i/5), alpha=.5, label='volatile')

            # if i == 0:
            #     ax2.legend()

        ax2.set_yticks([1,2,3,4,5,7,8,9,10,11,13,14,15,16,17])
        ax2.set_yticklabels([ '7','','', '3','', '.01', '','','','.05','', '1','','', '5'])
        # leg = ax2.get_legend()
        # leg.legendHandles[3].set_color('yellow')
        ax2.set_ylabel('\n c [days]\nt [rate]\nh [days]', linespacing=3, ha='center',
                       rotation='horizontal', labelpad=40, position=(0,.35))
        ax.set_ylabel("Microsoft Stock Price [$]", fontsize=24, labelpad=10)
        ax.set_xlabel("Time", fontsize=24, labelpad=10)
        ax.set_title("Microsoft Stock Inefficiencies")
        ax.set_ylim(295, 350)
        ax2.set_ylim(0,62)
        ax.tick_params(axis= 'x', which='major', labelsize=20,rotation=45)
        ax.tick_params(axis= 'y', which='major', labelsize=20)
        fig.tight_layout()
        plt.show()

        return
    
def inefficiencies(df, window_period, threshold_rate, confirmation_window=None, 
                       round_up_week=False):

    start = df.index

    if round_up_week:
        rest_of_week = (pd.Timedelta(weeks=1) - 
                        pd.Series([(i.dayofweek+1) * pd.Timedelta(days=1) for i in start]))
        window_period = window_period + rest_of_week

    end = start + window_period

    horizonmax = [max(df['high'][(df.index >= s) & (df.index < e)])
                  for s, e in zip(start, end)]
    horizonmin = [min(df['low'][(df.index >= s) & (df.index < e)])
                  for s, e in zip(start, end)]

    underpriced = (horizonmax/df['close'] - 1) > threshold_rate
    overpriced = (horizonmin/df['close'] - 1) < -threshold_rate
    efficient = ~(overpriced | underpriced)
    volatile = underpriced & overpriced

    if confirmation_window:
        underpriced = underpriced.rolling(window=confirmation_window).apply(lambda x: x.all())
        overpriced = overpriced.rolling(window=confirmation_window).apply(lambda x: x.all())
        efficient = efficient.rolling(window=confirmation_window).apply(lambda x: x.all())
        volatile = volatile.rolling(window=confirmation_window).apply(lambda x: x.all())

    data = np.arange(len(df))

    data[underpriced.astype(bool)] = 0
    data[overpriced.astype(bool)] = 1
    data[efficient.astype(bool)] = 2
    data[volatile.astype(bool)] = 3

    return pd.DataFrame(data=data, index=df.index, columns=['inefficiency'])




def find_corrs():
    
    path = os.getcwd()
    quiver_files = glob.glob(os.path.join(path, "data/quiver/*.csv"))

    desired_cols = {
        'gov_contracts' : ['Amount'],
        'house_trading' : ['Representative', 'Transaction', 'Amount'],
        'insiders' : ['Name', 'AcquiredDisposedCode', 'TransactionCode', 'Shares', 'PricePerShare', 'SharesOwnedFollowing'],
        'lobbying' : ['Amount'],
        'offexchange' : ['OTC_Short', 'OTC_Total', 'DPI'],
        'patents.csv' : ['Claims'],
        'sec13FChanges' : ['Change'],
        'senate_trading' : ['Senator', 'Transaction', 'Amount'],
        'spacs' : ['Mentions', 'Rank', 'Sentiment'],
        'twitter' : ['Followers', 'pct_change_day', 'pct_change_week', 'pct_change_month'],
        'wallstreetbets' :  ['Mentions', 'Rank', 'Sentiment'],
        "wikipedia" : ['Views', 'pct_change_week', 'pct_change_month']
    }

    to_join = []
    for f in quiver_files:
        try:
            cols = desired_cols[f[56:-4]]
            prefix = f[56:-4] + '_'
            to_join.append(pd.read_csv(f, index_col='Date', parse_dates=['Date'])[cols].add_prefix(prefix))
        except KeyError:
            print('ditched', f[56:-4])

    df = to_join[1].join(to_join[2:], how='outer').loc['2020-01-02' : '2020-12-31']
    
    dates = df.index 
    
    df2 = pd.read_csv("data/alpha/MSFT.csv", index_col='time', parse_dates=['time'])
    
    auto = []
    for date in dates:
        next_day = date + pd.Timedelta(days=1)
        day_before = date - pd.Timedelta(days=1)
        alpha_data = df2[(df2.index > date)&(df2.index < next_day)]
        day_before_data = df2[(df2.index > day_before)&(df2.index < date)]
        auto_correlate = alpha_data/day_before_data
        auto.append(auto_correlate)
        
    df['auto'] = auto 
    
    print(df)

    # df3 = df2.join(df, how='outer')

    # df3.to_csv('test10.csv')
    
    # print(df.join(df2, how='outer'))
   
    return

# find_corrs()


def preprocess():

    path = os.getcwd()
    quiver_files = glob.glob(os.path.join(path, "data/quiver/*.csv"))

    desired_cols = {
        'gov_contracts' : ['Amount'],
        'house_trading' : ['Representative', 'Transaction', 'Amount'],
        'insiders' : ['Name', 'AcquiredDisposedCode', 'TransactionCode', 'Shares', 'PricePerShare', 'SharesOwnedFollowing'],
        'lobbying' : ['Amount'],
        'offexchange' : ['OTC_Short', 'OTC_Total', 'DPI'],
        'patents.csv' : ['Claims'],
        'sec13FChanges' : ['Change'],
        'senate_trading' : ['Senator', 'Transaction', 'Amount'],
        'spacs' : ['Mentions', 'Rank', 'Sentiment'],
        'twitter' : ['Followers', 'pct_change_day', 'pct_change_week', 'pct_change_month'],
        'wallstreetbets' :  ['Mentions', 'Rank', 'Sentiment'],
        "wikipedia" : ['Views', 'pct_change_week', 'pct_change_month']
    }

    to_join = []
    for f in quiver_files:
        try:
            cols = desired_cols[f[56:-4]]
            prefix = f[56:-4] + '_'
            to_join.append(pd.read_csv(f, index_col='Date', parse_dates=['Date'])[cols].add_prefix(prefix))
        except KeyError:
            print('ditched', f[56:-4])


    df = to_join[1].join(to_join[2:], how='outer').loc['2020-01-02' : '2020-12-31']
    print(df.columns)
    
    onehot_ftrs = ['SEX']
    minmax_ftrs = ['wallstreetsbets_Rank']
    std_ftrs = ['wallstreetsbets_Mentions', 'sec13FChanges_Change', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6']

#      ['sec13FChanges_Change', 'wallstreetbets_Mentions',
#        'wallstreetbets_Rank', 'wallstreetbets_Sentiment',
#        'senate_trading_Senator', 'senate_trading_Transaction',
#        'senate_trading_Amount', 'house_trading_Representative',
#        'house_trading_Transaction', 'house_trading_Amount', 'lobbying_Amount',
#        'twitter_Followers', 'twitter_pct_change_day',
#        'twitter_pct_change_week', 'twitter_pct_change_month',
#        'offexchange_OTC_Short', 'offexchange_OTC_Total', 'offexchange_DPI',
#        'wikipedia_Views', 'wikipedia_pct_change_week',
#        'wikipedia_pct_change_month', 'gov_contracts_Amount', 'spacs_Mentions',
#        'spacs_Rank', 'spacs_Sentiment']

# preprocess()

    # preprocessor = ColumnTransformer(
    #     transformers=[
    #         ('onehot', OneHotEncoder(sparse=False,handle_unknown='ignore'), onehot_ftrs),
    #         ('minmax', MinMaxScaler(), minmax_ftrs),
    #         ('std', StandardScaler(), std_ftrs)])

    # clf = Pipeline(steps=[('preprocessor', preprocessor)]) 

    # plt.bar(x=all_but_wsb['name'], height=all_but_wsb['n_rows'], color=list('rgbkymc'))
    # plt.xticks(rotation=45, ha='right')
    # plt.yticks(fontsize=12)
    # plt.ylabel('Number of Rows', fontsize=14)
    # plt.xlabel('Dataset Name', fontsize=14)
    # plt.title('Quiver: Size of Each Dataset', fontsize=16)
    # plt.tight_layout()

    # plt.show()


    # # gathering metadata of the whole dataset
    # all_tickers = list(itertools.chain.from_iterable(
    #     [df_info[key]["tickers"] for key in df_info]))
    # all_columns = list(itertools.chain.from_iterable(
    #     [df_info[key]["columns"] for key in df_info]))
    # column_sets = [set(df_info[key]["columns"]) for key in df_info]

    # dataset_info = {
    #     "n_all_tickers" : len(all_tickers),
    #     "n_all_columns" : len(all_columns),
    #     "n_unique_tickers" : len(set(all_tickers)),
    #     "n_unique_columns" : len(set(all_columns)),
    #     "ubiquitous_columns" : column_sets[0].intersection(*column_sets)
    # }

    # _ = [print("shape of", key, ":", df_info[key]['shape']) for key in df_info]
    # _ = [print("ticker counts of", key, ":", df_info[key]['ticker_counts']) for key in df_info]
    # _ = [print(key, ":", val) for key, val in dataset_info.items()]




# datasets_items = datasets.items()
# name, dataset = datasets.items()[0]
# columns = ["", 'Date'] + [name + '_' + col for col in dataset.columns][2:]
# df = pd.DataFrame(data=dataset, columns=columns, index='Date')

# for name, ds in datasets.items()[1:]:
#     columns = ["", 'Date'] + [name + '_' + col for col in ds.columns][2:]
#     print(columns)
#     df.merge(ds)

# print(df.head())
# print(datasets.items())
# for key,val in datasets.items():
#     print(key, val.head())
#     # df[key] = val

api_time = {
    'alpha' : 'time',
    'quiver' : 'Date'
}

alpha = pd.read_csv("data/alpha/MSFT.csv", index_col=api_time['alpha'], parse_dates=[api_time['alpha']])



# to_join = []

# for f in quiver_files:
#     try:
#         cols = desired_cols[f[56:-4]]
#         prefix = f[56:-4] + '_'
#         to_join.append(pd.read_csv(f, index_col='Date', parse_dates=['Date'])[cols].add_prefix(prefix))
#         # print("adding", f[56:-4])
#     # except ValueError:
#     #     pass
#     except KeyError:
#         print('ditched', f[56:-4])


# df0 = to_join[0]

# for df in to_join[1:]:
#     try:
#         df0.join(df, in_place=True)
#     except TypeError:
#         # print(df)
#         pass
# print(to_join)


# house_trading = pd.read_csv("data/quiver/house_trading.csv", 
#     index_col=api_time['quiver'])[desired_cols['house_trading']].add_prefix('house_trading_')

# senate_trading = pd.read_csv("data/quiver/senate_trading.csv", 
#     index_col=api_time['quiver'])[desired_cols['senate_trading']].add_prefix('senate_trading_')

# gov_contracts = pd.read_csv("data/quiver/gov_contracts.csv", 
#     index_col=api_time['quiver'])[desired_cols['gov_contracts']].add_prefix('gov_contracts_')

# print(house_trading.head())
# print(senate_trading.head())
# print(gov_contracts.head())


# print(min(alpha['time']))
# print(min(house_trading['Date']))
# print(max(alpha['time']))
# print(max(house_trading['Date'])) 

# start_date = max(min(alpha['time']), min(house_trading['Date']))

# print(start_date)
# df['quiver_time'] = house_trading['Date']
# df['alpha_time'] = alpha['time']

# for column in df.columns:
    # print(df[column].shape)
    # print(df[column].value_counts())


# print(len(house_trading))
# print(len(senate_trading))
# print(len(gov_contracts))
# df3 = pd.DataFrame()
# df3['Date'] = pd.date_range('2020-1-1', periods=365, freq='D')

# df = to_join[1].join(to_join[2:], how='outer').loc['2020-01-02' : '2020-12-31']

# print(df.columns) # no insider's

# print(len(df.loc['2020-01-02' : '2021-01-01']))
# print(df.loc['2020-01-02' : '2021-01-01'].resample('D').sum().dtypes)

# df.loc['2021-01-02' : '2022-01-01'].to_csv('Test.csv')
# df.loc['2021-01-02' : '2022-01-01'].resample('D').sum().ffill(axis=).to_csv('Test2.csv')

def candlesticks(df):

    plt.figure(figsize=[24,5])

    width = sqrt(sqrt(len(df)))*.004
    width2 = width/6.2

    up = df[df['close'] >= df['open']]
    down = df[df['close'] < df['open']]

    up_color = 'green'
    down_color = 'red'

    plt.bar(up.index, up.close - up.open, width, bottom = up.open, color=up_color)
    plt.bar(up.index, up.high - up.close, width2, bottom = up.close, color=up_color)
    plt.bar(up.index, up.open - up.low, width2, bottom = up.low, color=up_color)

    plt.bar(down.index, down.open - down.close, width, bottom = down.close, color=down_color)
    plt.bar(down.index, down.high - down.open, width2, bottom = down.open, color=down_color)
    plt.bar(down.index, down.close - down.low, width2, bottom = down.low, color=down_color)

    plt.xticks(rotation=45, ha='right')

    return


df2 = alpha.loc[(alpha.index >= '2021-01-01') & (alpha.index < '2022-01-01')].copy()
df2['change'] = df2['open'] - df2['close']
# df2['change'][:20].plot.line()
# plt.show()
df2['range'] = df2['high'] - df2['low']
df2['change/range'] = df2['change'] / df2['range']

# df2['change/range'][:20].plot.line()
# plt.show()
# candlesticks(df2.iloc[:20])
# class WeekendHorizonIndexer(BaseIndexer):

#     def __init__(self, start_date, period, window_size, indexx):
#         super().__init__(window_size=window_size)
#         self.start_date = start_date
#         self.period = period
#         self.indexx = indexx

#     def get_window_bounds(self, num_values, min_periods, center, closed):

#         start_day_of_week = self.start_date.dayofweek
#         indexx_index = np.arange(len(self.indexx))

#         print(self.start_date, start_day_of_week)
#         print(self.indexx)
#         start = np.empty(num_values, dtype=np.int64)
#         end = np.empty(num_values, dtype=np.int64)
#         for i in np.arange(num_values):
#             t = self.start_date + (i * self.period)
#             t_end = t + pd.Timedelta(weeks=1) + (7 - t.dayofweek)*pd.Timedelta(days=1)
#             in_range = (self.indexx < t_end)
#             # print(sum(in_range))
#             start[i] = i
#             end[i] = max(indexx_index[in_range])

#         return start, end

# class WeekendHorizonIndexer(BaseIndexer):

#     def get_window_bounds(self, num_values, min_periods, center, closed):
#         start = np.empty(num_values, dtype=np.int64)
#         end = np.empty(num_values, dtype=np.int64)
#         for i in np.arange(num_values):
#             # print(sum(in_range))
#             start[i] = i
#             end[i] = i + ((32*7) - i % (32*7))

#         return start, end
# # testy = WeekendHorizonIndexer(start_date = df2.index[0], period = pd.Timedelta(minutes=30), window_size=1, indexx = df2.index)

# candlesticks(df2.iloc[:200].rolling(window=7).sum())



# df2['horizonmax'] = [max(df2['high'][(df2.index >= s) & (df2.index < e)]) for s, e in zip(start, ends)]
# df2['inefficient'] = (df2['horizonmax']/df2['close'] - 1) > .05
# df2.to_csv('test8.csv')

# candlesticks(df2.iloc[:200]) 
# candlesticks(df2.iloc[:200].rolling(window=7).sum())



def candlesticky(df):
    
    fig, ax = plt.subplots(figsize=[24,5])

    width = sqrt(sqrt(len(df)))*.004
    width2 = width/6.2

    up = df[df['close'] >= df['open']]
    down = df[df['close'] < df['open']]

    up_color = 'green'
    down_color = 'red'

    plt.bar(up.index, up.close - up.open, width, bottom = up.open, color=up_color)
    plt.bar(up.index, up.high - up.close, width2, bottom = up.close, color=up_color)
    plt.bar(up.index, up.open - up.low, width2, bottom = up.low, color=up_color)

    plt.bar(down.index, down.open - down.close, width, bottom = down.close, color=down_color)
    plt.bar(down.index, down.high - down.open, width2, bottom = down.open, color=down_color)
    plt.bar(down.index, down.close - down.low, width2, bottom = down.low, color=down_color)
    
    sx2 = ax.twinx().set_ylim(0,10)
    
    underpriced = df[df['inefficiencies'] == 0]
    overpriced = df[df['inefficiencies'] == 1]
    efficient = df[df['inefficiencies'] == 2]
    volatile = df[df['inefficiencies'] == 3]

    plt.bar(volatile.index,height=1,width=1.4*width, color='yellow', bottom=0)
    plt.bar(underpriced.index,height=1,width=1.4*width, color='green', bottom=0)
    plt.bar(overpriced.index,height=1,width=1.4*width, color='red', bottom=0)
    plt.bar(efficient.index,height=1,width=1.4*width, color='blue', bottom=0)

    plt.xticks(rotation=45, ha='right')

    plt.show()

    return


def candlesticky2(df):
    
    fig, ax = plt.subplots(figsize=[24,5])

    width = sqrt(sqrt(len(df)))*.004
    width2 = width/6.2

    up = df[df['close'] >= df['open']]
    down = df[df['close'] < df['open']]

    up_color = 'green'
    down_color = 'red'

    plt.bar(up.index, up.close - up.open, width, bottom = up.open, color=up_color)
    plt.bar(up.index, up.high - up.close, width2, bottom = up.close, color=up_color)
    plt.bar(up.index, up.open - up.low, width2, bottom = up.low, color=up_color)

    plt.bar(down.index, down.open - down.close, width, bottom = down.close, color=down_color)
    plt.bar(down.index, down.high - down.open, width2, bottom = down.open, color=down_color)
    plt.bar(down.index, down.close - down.low, width2, bottom = down.low, color=down_color)
    
    sx2 = ax.twinx()
    
    for i in range(5):

        underpriced = df[df['inefficiency' + str(i)] == 0]
        overpriced = df[df['inefficiency' + str(i)] == 1]
        efficient = df[df['inefficiency' + str(i)] == 2]
        volatile = df[df['inefficiency' + str(i)] == 3]

        plt.bar(volatile.index, height=1, width=14*width, color='yellow', bottom=i, alpha=.01)
        plt.bar(underpriced.index, height=1, width=14*width, color='green', bottom=i, alpha=.01)
        plt.bar(overpriced.index, height=1, width=14*width, color='red', bottom=i, alpha=.01)
        #plt.bar(efficient.index,height=.2,width=1.4*width, color='blue', bottom=i*.2)
    

    plt.xticks(rotation=45, ha='right')

    plt.show()

    return





# ineffy = [inefficiencies(df2, window_period=pd.Timedelta(days=7), threshold_rate=0.05, round_up_week=True).add_suffix('0'),
#     inefficiencies(df2, window_period=pd.Timedelta(days=6), threshold_rate=0.05, round_up_week=True).add_suffix('1'),
#     inefficiencies(df2, window_period=pd.Timedelta(days=5), threshold_rate=0.05, round_up_week=True).add_suffix('2'),
#     inefficiencies(df2, window_period=pd.Timedelta(days=4), threshold_rate=0.05, round_up_week=True).add_suffix('3'),
#     inefficiencies(df2, window_period=pd.Timedelta(days=3), threshold_rate=0.05, round_up_week=True).add_suffix('4')]

# ineffy2 = [inefficiencies(df2, window_period=pd.Timedelta(days=7), threshold_rate=0.05, round_up_week=True).add_suffix('0'),
#     inefficiencies(df2, window_period=pd.Timedelta(days=7), threshold_rate=0.04, round_up_week=True).add_suffix('1'),
#     inefficiencies(df2, window_period=pd.Timedelta(days=7), threshold_rate=0.03, round_up_week=True).add_suffix('2'),
#     inefficiencies(df2, window_period=pd.Timedelta(days=7), threshold_rate=0.02, round_up_week=True).add_suffix('3'),
#     inefficiencies(df2, window_period=pd.Timedelta(days=7), threshold_rate=0.01, round_up_week=True).add_suffix('4')]

# ineffy3 = [inefficiencies(df2, window_period=pd.Timedelta(days=7), threshold_rate=0.01, confirmation_window=pd.Timedelta(days=1), round_up_week=True).add_suffix('0'),
#     inefficiencies(df2, window_period=pd.Timedelta(days=7), threshold_rate=0.01, confirmation_window=pd.Timedelta(days=2), round_up_week=True).add_suffix('1'),
#     inefficiencies(df2, window_period=pd.Timedelta(days=7), threshold_rate=0.01, confirmation_window=pd.Timedelta(days=3), round_up_week=True).add_suffix('2'),
#     inefficiencies(df2, window_period=pd.Timedelta(days=7), threshold_rate=0.01, confirmation_window=pd.Timedelta(days=4), round_up_week=True).add_suffix('3'),
#     inefficiencies(df2, window_period=pd.Timedelta(days=7), threshold_rate=0.01, confirmation_window=pd.Timedelta(days=5), round_up_week=True).add_suffix('4')]


# for i in ineffy:
#     print(i.value_counts())

# df2 = df2.join(ineffy3, how='outer')

# print(df2.iloc[:200])
# # df2.rolling(window=224).sum().to_csv('test5.csv')

# # df2 = df2.ffill()
# # print(df2.iloc[:200])
# candlesticky2(df2.iloc[:2000])

# df3 = df2[:1000].copy()
# ineffy4 = [inefficiencies(df3, window_period=pd.Timedelta(days=7), threshold_rate=0.05).add_suffix('0'),
#     inefficiencies(df3, window_period=pd.Timedelta(days=6), threshold_rate=0.05).add_suffix('1'),
#     inefficiencies(df3, window_period=pd.Timedelta(days=5), threshold_rate=0.05).add_suffix('2'),
#     inefficiencies(df3, window_period=pd.Timedelta(days=4), threshold_rate=0.05).add_suffix('3'),
#     inefficiencies(df3, window_period=pd.Timedelta(days=3), threshold_rate=0.05).add_suffix('4')]

# ineffy5 = [inefficiencies(df3, window_period=pd.Timedelta(days=7), threshold_rate=0.01).add_suffix('5'),
#     inefficiencies(df3, window_period=pd.Timedelta(days=7), threshold_rate=0.02).add_suffix('6'),
#     inefficiencies(df3, window_period=pd.Timedelta(days=7), threshold_rate=0.03).add_suffix('7'),
#     inefficiencies(df3, window_period=pd.Timedelta(days=7), threshold_rate=0.04).add_suffix('8'),
#     inefficiencies(df3, window_period=pd.Timedelta(days=7), threshold_rate=0.05).add_suffix('9')]

# ineffy6 = [inefficiencies(df3, window_period=pd.Timedelta(days=7), threshold_rate=0.01, confirmation_window=pd.Timedelta(days=1), round_up_week=True).add_suffix('10'),
#     inefficiencies(df3, window_period=pd.Timedelta(days=7), threshold_rate=0.01, confirmation_window=pd.Timedelta(days=2), round_up_week=True).add_suffix('11'),
#     inefficiencies(df3, window_period=pd.Timedelta(days=7), threshold_rate=0.01, confirmation_window=pd.Timedelta(days=3), round_up_week=True).add_suffix('12'),
#     inefficiencies(df3, window_period=pd.Timedelta(days=7), threshold_rate=0.01, confirmation_window=pd.Timedelta(days=4), round_up_week=True).add_suffix('13'),
#     inefficiencies(df3, window_period=pd.Timedelta(days=7), threshold_rate=0.01, confirmation_window=pd.Timedelta(days=5), round_up_week=True).add_suffix('14')]


def candlesticky3(df):
    
    for i in range(15):
        try:
            assert('inefficiency'+str(i) in df.columns)
        except AssertionError:
            print('this visualization requires 15 inefficiency columns')
    
    fig, ax = plt.subplots(figsize=[24,5])

    width = sqrt(sqrt(len(df)))*.004
    width2 = width/6.2

    up = df[df['close'] >= df['open']]
    down = df[df['close'] < df['open']]

    up_color = 'green'
    down_color = 'red'

    plt.bar(up.index, up.close - up.open, width, bottom = up.open, color=up_color)
    plt.bar(up.index, up.high - up.close, width2, bottom = up.close, color=up_color)
    plt.bar(up.index, up.open - up.low, width2, bottom = up.low, color=up_color)

    plt.bar(down.index, down.open - down.close, width, bottom = down.close, color=down_color)
    plt.bar(down.index, down.high - down.open, width2, bottom = down.open, color=down_color)
    plt.bar(down.index, down.close - down.low, width2, bottom = down.low, color=down_color)

    ax2 = ax.twinx()

    for i in range(15):

        underpriced = df[df['inefficiency' + str(i)] == 0]
        overpriced = df[df['inefficiency' + str(i)] == 1]
        efficient = df[df['inefficiency' + str(i)] == 2]
        volatile = df[df['inefficiency' + str(i)] == 3]

        ax2.bar(underpriced.index, height=.95, width=width, color='green', bottom=0.5 + i + floor(i/5), alpha=.5, label= 'underpriced')
        ax2.bar(overpriced.index, height=.95, width=width, color='red', bottom=0.5 + i + floor(i/5), alpha=.5, label= 'overpriced')
        ax2.bar(efficient.index,height=.95, width=width, color='blue', bottom=0.5 + i + floor(i/5),alpha=.5, label= 'efficient')
        ax2.bar(volatile.index, height=.95, width=width, color='yellow', bottom=0.5 + i + floor(i/5), alpha=.5, label='volatile')

        # if i == 0:
        #     ax2.legend()

    ax2.set_yticks([1,2,3,4,5,7,8,9,10,11,13,14,15,16,17])
    ax2.set_yticklabels([ '7','','', '3','', '.01', '','','','.05','', '1','','', '5'])
    # leg = ax2.get_legend()
    # leg.legendHandles[3].set_color('yellow')
    ax2.set_ylabel('\n c [days]\nt [rate]\nh [days]', linespacing=3, ha='center', rotation='horizontal', labelpad=40, position=(0,.35))
    ax.set_ylabel("Microsoft Stock Price [$]", fontsize=24, labelpad=10)
    ax.set_xlabel("Time", fontsize=24, labelpad=10)
    ax.set_title("Microsoft Stock Inefficiencies")
    ax.set_ylim(295, 350)
    ax2.set_ylim(0,62)
    ax.tick_params(axis= 'x', which='major', labelsize=20,rotation=45)
    ax.tick_params(axis= 'y', which='major', labelsize=20)
    fig.tight_layout()
    plt.show()

    return


# df2 = df2.ffill()
# print(df2.iloc[:200])

# df3 = df3.join([*ineffy4, *ineffy5, *ineffy6], how='outer')

# print(df3.iloc[:200])

# candlesticky3(df3)


