#!/usr/bin/env python
# coding: utf-8

# # Working with two or many Strategies (Combination)

# ## Introduction and Data

# Measures to reduce Trading Costs:
# - Busy Trading Hours
# - The right Granularity
# - Better/more complex Strategies with stronger Signals -> go/stay neutral if signals are weak

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8")


# In[ ]:


df = pd.read_csv("twenty_minutes.csv", parse_dates = ["time"], index_col = "time")
df


# In[ ]:


df.info()


# In[ ]:





# ## SMA Strategy

# In[11]:


import SMABacktester as SMA


# In[12]:


start = "2019-01-01"
end = "2020-08-30"
symbol = "EURUSD"
sma_s = 50
sma_l = 150
tc = 0.00007


# In[15]:


tester = SMA.SMABacktester( symbol ,sma_s, sma_l, start, end, tc)


# In[ ]:


tester


# In[ ]:


tester.data


# In[ ]:


tester.test_strategy()


# In[ ]:


tester.plot_results()


# In[ ]:


tester.results


# In[ ]:


tester.results[["SMA_S", "SMA_L", "position"]].plot(secondary_y = "position", figsize = (12, 8))
plt.show()


# In[ ]:


tester.results.trades.value_counts()


# In[ ]:





# ## Mean Reversion Strategy

# In[ ]:


import MeanRevBacktester as MeanRev


# In[ ]:


start = "2019-01-01"
end = "2020-08-30"
symbol = "EURUSD"
sma = 75
dev = 3
tc = 0.00007


# In[ ]:


tester2 = MeanRev.MeanRevBacktester(symbol, sma, dev, start, end, tc)


# In[ ]:


tester2


# In[ ]:


tester2.data


# In[ ]:


tester2.test_strategy()


# In[ ]:


tester2.plot_results()


# In[ ]:


tester2.results


# In[ ]:


tester2.results.position.plot(figsize = (12, 8))
plt.show()


# In[ ]:


tester2.results.trades.value_counts()


# In[ ]:





# ## Combining both Strategies

# __Goal:__ Stronger Signals / Identify Weak Signals

# __Two different Methods:__

# __Unanimous Signals__ (pro: strong signals | con: restrictive / doesn´t work with too many Indicators)
# - Go Long if all Signals are long
# - Go Short if all Signals are short
# - Go Neutral if Signals are nonunanimous 

# __Majority / Tendency__ (pro: can be customized | con: more trades / weaker signals)
# - Go Long if at least two Signals are long (3 Signals Case)
# - Go Long if > [50%] of Signals are long and < [25%] of Signals are short (many Signals Case)

# In[ ]:


tester.results


# In[ ]:


tester2.results


# In[ ]:


comb = tester.results.loc[:, ["returns", "position"]].copy()


# In[ ]:


comb


# In[ ]:


comb.rename(columns = {"position":"position_SMA"}, inplace = True)


# In[ ]:


comb["position_MR"] = tester2.results.position.astype("int")


# In[ ]:


comb


# __Alternative 1: Unanimous Signals__

# In[ ]:


#comb["position_comb"] = np.where(comb.position_MR == comb.position_SMA, comb.position_MR, 0)


# __Alternative 2: Majority / Tendency__

# In[ ]:


comb["position_comb"] = np.sign(comb.position_MR + comb.position_SMA)


# In[ ]:


comb.head(60)


# In[ ]:


comb.position_comb.value_counts()


# In[ ]:


comb.position_comb.plot(figsize = (12, 8))
plt.show()


# In[ ]:





# ## Taking into account busy trading hours

# In[ ]:


comb


# In[ ]:


comb["NYTime"] = comb.index.tz_convert("America/New_York")
comb["hour"] = comb.NYTime.dt.hour


# In[ ]:


comb.position_comb = np.where(comb.hour.between(2, 12), comb.position_comb, 0)


# In[ ]:


comb.position_comb.value_counts()


# In[ ]:


comb.position_comb.plot(figsize = (12, 8))
plt.show()


# In[ ]:


comb.position_comb.loc["2020-08"].plot(figsize = (12, 8))
plt.show()


# In[ ]:





# ## Backtesting

# In[ ]:


comb


# In[ ]:


comb["strategy"] = comb["position_comb"].shift(1) * comb["returns"]


# In[ ]:


comb.dropna(inplace=True)


# In[ ]:


comb["trades"] = comb.position_comb.diff().fillna(0).abs()


# In[ ]:


tc = 0.000059


# In[ ]:


comb.strategy = comb.strategy - comb.trades * tc


# In[ ]:


comb["creturns"] = comb["returns"].cumsum().apply(np.exp)
comb["cstrategy"] = comb["strategy"].cumsum().apply(np.exp)


# In[ ]:


comb[["creturns", "cstrategy"]].plot(figsize = (12, 8), title = "EUR/USD - SMA{} | SMA{}".format(sma_s, sma_l), fontsize = 12)
plt.legend(fontsize = 12)
plt.show()


# In[ ]:


comb.trades.value_counts()


# In[ ]:





# ## Strategy Optimization

# In[ ]:


import SMABacktester as SMA
import MeanRevBacktester as MeanRev


# In[ ]:


def optimal_strategy(parameters):
    
    start = "2019-01-01"
    end = "2020-08-30"
    symbol = "EURUSD"
    tc = 0.000059
    
    # SMA
    tester1 = SMA.SMABacktester(symbol, int(parameters[0]), int(parameters[1]), start, end, tc)
    tester1.test_strategy()
    
    # Bollinger
    tester2 = MeanRev.MeanRevBacktester(symbol,  int(parameters[2]),  int(parameters[3]), start, end, tc)
    tester2.test_strategy()
    
    # Create comb
    comb = tester1.results.loc[:, ["returns", "position"]].copy()
    comb.rename(columns = {"position":"position_SMA"}, inplace = True)
    comb["position_MR"] = tester2.results.position
    
    # 2 Methods
    #comb["position_comb"] = np.where(comb.position_MR == comb.position_SMA, comb.position_MR, 0) 
    comb["position_comb"] = np.sign(comb.position_MR + comb.position_SMA)
    
    # Busy Hours
    comb["NYTime"] = comb.index.tz_convert("America/New_York")
    comb["hour"] = comb.NYTime.dt.hour
    comb.position_comb = np.where(comb.hour.between(2, 12), comb.position_comb, 0)
    
    # Backtest
    comb["strategy"] = comb["position_comb"].shift(1) * comb["returns"]
    comb.dropna(inplace=True)
    comb["trades"] = comb.position_comb.diff().fillna(0).abs()
    comb.strategy = comb.strategy - comb.trades * tc
    comb["creturns"] = comb["returns"].cumsum().apply(np.exp)
    comb["cstrategy"] = comb["strategy"].cumsum().apply(np.exp)
    
    return -comb["cstrategy"].iloc[-1] # negative absolute performance to be minimized


# In[ ]:


optimal_strategy((50, 150, 75, 3))


# In[ ]:


from scipy.optimize import minimize


# In[ ]:


bnds =  ((25, 75), (100, 200), (50, 100), (1, 5))
bnds


# In[ ]:


start_par = (50, 150, 75, 3)


# In[ ]:


#run optimization based on function to be minimized, starting with start parameters
opts = minimize(optimal_strategy, start_par, method = "Powell" , bounds = bnds)


# In[ ]:


opts


# In[ ]:


optimal_strategy(opts.x)


# In[ ]:




