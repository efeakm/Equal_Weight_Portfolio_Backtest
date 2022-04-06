
import os
os.chdir(os.getcwd())
import pandas as pd
import numpy as np
import pandas_datareader as pdr
import matplotlib.pyplot as plt
import time
import streamlit as st
plt.style.use('seaborn')

st.header('Equal Weight Portfolio')


###INPUT PARAMETERS
#========================================================
number_of_input_tickers_1 = st.slider('Number of stocks for portfolio 1', max_value = 10)

tickers = []
for i in range(number_of_input_tickers_1):
    tickers.append(st.text_input(f'Stock {i+1}'))
    
st.write('Selected Tickers')
st.write(tickers)


tickers_2 = ['SPY']


form = st.form('input_form')

start = form.date_input('Start Date', value = pd.to_datetime('2018-01-01'))
end = form.date_input('End Date', value = pd.to_datetime('2021-01-01'))




portfolio_value = form.number_input('Initial Amount', value = 100.0000,
                                  help ='please enter initial amount')

REINVESTMENT_OPTION = form.radio('Reinvestment Option', ['percentage','constant'])

if REINVESTMENT_OPTION == 'constant':
    REINVEST_AMOUNT = form.number_input('Constant reinvestment amount', value = 20.00)

REINVEST_PERCENTAGE = form.number_input('Reinvestment Percentage', value = 10.00)


BELOW_INITIAL_PERCENTAGE = 50

WEEKLY_INVESTMENT_AMOUNT = form.number_input('Weekly Investment Amount', value = 10.00)

MONTHLY_INVESTMENT_AMOUNT = form.number_input('Monthly Investment Amount', value = 50.00)

runn = form.form_submit_button('Run backtesting')






### MAIN FUNCTION
#========================================================================
@st.cache(suppress_st_warning=True)
def main_func(temp_input, tickers):
    
    
    ###INITIALIZATION
    #=========================================================================
    temp = temp_input.copy()
    no_of_stocks = len(tickers)

    
    #create columns
    for ticker in tickers:
        temp[f'{ticker}_size'] = 0
    temp['total_value'] = 0 
    
    
    #First buy orders
    for ticker in tickers:
        temp.loc[0,f'{ticker}_size'] = portfolio_value/(no_of_stocks * temp.loc[0,f'Adj Close_{ticker}'])
    temp.loc[0,'total_value'] = portfolio_value
    
    

    
    ###PERIODIC REINVESTMENT
    #=============================================================================
    
    temp['periodic_reinvestment'] = 0
    
    #invest every week
    if WEEKLY_INVESTMENT_AMOUNT > 0:
        
        
        temp['Date_temp'] = temp['Date'].dt.year.astype('str')+ '-' + temp['Date'].dt.isocalendar().week.astype('str')
        
        mask = temp.groupby('Date_temp').first()['Date']
        mask = temp['Date'].isin(mask)
        
        temp.loc[mask,'periodic_reinvestment'] = temp.loc[mask,'periodic_reinvestment'] + WEEKLY_INVESTMENT_AMOUNT
        
        temp = temp.drop('Date_temp',axis=1)
        temp.iloc[0,-1] = 0
        
        
    #invest every month    
    if MONTHLY_INVESTMENT_AMOUNT > 0:
        
        temp['Date_temp'] = temp['Date'].dt.year.astype('str')+ '-' + temp['Date'].dt.month.astype('str')
        
        mask = temp.groupby('Date_temp').first()['Date']
        mask = temp['Date'].isin(mask)
        
        temp.loc[mask,'periodic_reinvestment'] = temp.loc[mask,'periodic_reinvestment'] + MONTHLY_INVESTMENT_AMOUNT
        
        temp = temp.drop('Date_temp',axis=1)
        temp.iloc[0,-1] = 0
        
    
    
    ###REINVESTMENT BELOW THRESHOLD AND MAIN LOGIC ON FOR LOOP
    #=========================================================================
    temp = temp.reset_index(drop=True)
    temp['reinvestment_due_to_loss'] = 0
    below_initial_multiplier = 1.0
    
    for i in range(1,len(temp)+1):
        
        #Previous days positions
        for ticker in tickers:
            temp.loc[i,f'{ticker}_size'] = temp.loc[i-1,f'{ticker}_size']
    
        #Total value calculation
        total_value = 0
        for ticker in tickers:
            total_value += temp.loc[i,f'{ticker}_size'] * temp.loc[i,f'Adj Close_{ticker}']
        temp.loc[i,'total_value'] = total_value 
        
        
        #If there is periodic reinvestment, update total_value and ticker sizes
        if temp.loc[i,'periodic_reinvestment'] != 0:
            temp.loc[i,'total_value'] = temp.loc[i,'total_value'] + temp.loc[i,'periodic_reinvestment']
            for ticker in tickers:
                temp.loc[i,f'{ticker}_size'] = ((temp.loc[i,'total_value']/no_of_stocks) /
                                                temp.loc[i,f'Adj Close_{ticker}'])
        
        
        #If portfolio suffers a loss below REINVEST_PERCENTAGE, reinvest
        top_value = temp.loc[:i,'total_value'].max()
        current_ratio = temp.loc[i,'total_value'] / top_value
        
        #When total_value passes top_value again, reset below_initial_multiplier
        if temp.loc[i,'total_value'] >= top_value:
            below_initial_multiplier = 1.0
            
        if current_ratio <= (1 - REINVEST_PERCENTAGE/100):
            
            if REINVESTMENT_OPTION == 'constant':
                
                reinv_amount = REINVEST_AMOUNT * below_initial_multiplier
                
                #if total_value falls below initial amount,
                #increase it by BELOW_INITIAL_PERCENTAGE
                if temp.loc[i,'total_value'] < portfolio_value:
                    below_initial_multiplier = (below_initial_multiplier * 
                                                (1 + BELOW_INITIAL_PERCENTAGE/100))
                    
                

            elif REINVESTMENT_OPTION == 'percentage':
                reinv_amount = top_value - temp.loc[i,'total_value']
            else:
                print('wrong REINVESTMENT_OPTION, it should be percentage or constant')
                time.sleep(5)
                
            temp.loc[i,'reinvestment_due_to_loss'] = reinv_amount
            
            #update total_value
            temp.loc[i,'total_value'] = temp.loc[i,'total_value'] + reinv_amount
            
            #update ticker sizes
            for ticker in tickers:
                temp.loc[i,f'{ticker}_size'] = ((temp.loc[i,'total_value']/no_of_stocks) /
                                                temp.loc[i,f'Adj Close_{ticker}'])
        
    

    
    
    
    ###CHANGES IN POSITION SIZE
    #=============================================================================
    for ticker in tickers:
        temp[f'{ticker}_change'] = temp[f'{ticker}_size'] - temp[f'{ticker}_size'].shift(1)
        temp[f'{ticker}_change'] = temp[f'{ticker}_change'] * temp[f'Adj Close_{ticker}']
        
        temp.loc[0,f'{ticker}_change'] = portfolio_value/no_of_stocks
        
        
        
    #Total value in terms of percentage
    temp['total_value_percentage'] = (temp['total_value'] * 100 / portfolio_value) - 100
    
        
    ###MAX DRAWDOWN FUNC
    #======================================================================
    def drawdown_func(df,window):
        
        # Calculate the max drawdown in the past window days for each day in the series.
        # Use min_periods=1 if you want to let the first window days data have an expanding window
        Roll_Max = df['total_value'].rolling(window, min_periods=1).max()
        Daily_Drawdown = (df['total_value']/Roll_Max - 1.0).to_frame()
        Daily_Drawdown.index = df['Date']
        
    
        
        #En uzun drawdown suresi
        Daily_Drawdown['duration'] = np.nan
        counter = 0
        
        for row in Daily_Drawdown.index:
            
            if Daily_Drawdown.loc[row,'total_value'] == 0:
                counter = 0
            else:
                counter = counter + 1
            
            Daily_Drawdown.loc[row,'duration'] = counter
            
        
        # Next we calculate the minimum (negative) daily drawdown in that window.
        # Again, use min_periods=1 if you want to allow the expanding window
        # Max_Daily_Drawdown = Daily_Drawdown.rolling(window, min_periods=1).min()
        
        return Daily_Drawdown
    
    
    temp['Drawdown'] = drawdown_func(temp,len(temp))['total_value'].values
    temp['Drawdown'] = 100 * temp['Drawdown']
    
    return temp



if runn:

    
    ###CREATE HISTORICAL PRICE DATASET
    #========================================================
    df = pdr.get_data_yahoo('MSFT', start,end)
    df_1 = pd.DataFrame(df['Adj Close'].copy())
    
    
    for TICKER in tickers:
        # start = pd.to_datetime("2018-01-01")
        # end = pd.to_datetime("2030-01-01")
        df = pdr.get_data_yahoo(TICKER, start,end)
        
        df_1 = df_1.join(df['Adj Close'],rsuffix=f'_{TICKER}')
    
    
    df_1 = df_1.drop('Adj Close',axis=1)
    df_1 = df_1.reset_index()
    
    temp = main_func(df_1,tickers)
    
    
    
    
    ###COMPARISON DATA
    #=========================================================================
    df = pdr.get_data_yahoo('SPY', start,end)
    df_2 = pd.DataFrame(df['Adj Close'].copy())
    no_of_stocks_2 = len(tickers_2)
    
    
    for TICKER in tickers_2:
        df = pdr.get_data_yahoo(TICKER, start,end)
        
        df_2 = df_2.join(df['Adj Close'],rsuffix=f'_{TICKER}')
    
    
    df_2 = df_2.drop('Adj Close',axis=1)
    df_2 = df_2.reset_index()
    
    df_2 = main_func(df_2,tickers_2)
    
    
    
  
    
    
    ###OUTPUT
    #=======================================================================
    temp = temp.drop(temp.index[-1], axis = 0)
    
    
    #decimal fix
    col_list = (temp.columns[temp.columns.str.contains('change')].to_list()
                + ['total_value','reinvestment_due_to_loss','total_value_percentage'])
    temp[col_list] = round(temp[col_list],2)
    
    
    #Denominate change in USD
    for ticker in tickers:
        temp[f'{ticker}_size'] = temp[f'{ticker}_size'] * temp[f'Adj Close_{ticker}']
    
    
    #Change column names
    cols = temp.columns[temp.columns.str.contains('Adj Close')]
        
    names = [f'Price_{ticker}' for ticker in tickers]
    
    mapp = {}
    for i in range(len(names)):
        mapp[cols[i]] = names[i]
    
    
    temp = temp.rename(mapp, axis=1)
    
    
    
    
    #Output file
    st.subheader('Output File')
    st.write(temp)
    
    


  
    ###VISUALIZATION
    #==================================================
    plt.style.use('seaborn')
    fig, ax1 = plt.subplots(figsize=(10,7))
    
    
    
    ax1.plot(temp['Date'], temp['Drawdown'], color = '#C70039', alpha = 0.2)
    ax1.fill_between(temp['Date'], temp['Drawdown'], 0, color = '#C70039', alpha = 0.2)
    
    
    ax1.set_ylabel('Drawdown', fontsize = 14)
    
    
    ax2 = ax1.twinx()
    
    ax2.plot(temp['Date'],temp['total_value'], label = 'Main_Portfolio')
    ax2.plot(df_2['Date'],df_2['total_value'], label = 'Comparison')
    
    ax2.legend(loc = 'lower right')
    ax2.set_ylabel('Total Portfolio Value', fontsize = 14)
    
    
    ax2.annotate('%0.2f' % temp.iloc[-2]['total_value'],
                  xy=(1, temp.iloc[-2]['total_value']), xytext=(8, 0), 
                  xycoords=('axes fraction', 'data'), textcoords='offset points')
    
    ax2.annotate('%0.2f' % df_2.iloc[-2]['total_value'],
                  xy=(1, df_2.iloc[-2]['total_value']), xytext=(8, 0), 
                  xycoords=('axes fraction', 'data'), textcoords='offset points')
    
    fig.tight_layout() 
    
    st.pyplot(fig)
    fig.savefig('comparison_plot.png')
    


    
    
    


    




    st.download_button('Save the output data', temp.to_csv(), file_name = 'output.csv')


    with open("comparison_plot.png", "rb") as file:
        btn = st.download_button(
                label="Download image",
                data=file,
                file_name="comparison_plot.png",
                mime="image/png",
                )
     

    
    
    
    