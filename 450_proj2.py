import csv
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

########################################################################################################################

#getting data ready for analysis

#read in csv with 100 random stocks
df = pd.read_csv(r"C:\Users\chada\OneDrive\Desktop\econ_450\stocks_data.csv")
#turn csv into list of lists
list_of_lists = df.values.tolist()

#create empty lists for different categories
dates = []
perm = []
sp_500 = []
tickers = []

#populating lists for dates, and permno
for i in range(60):
    dates.append(list_of_lists[i][1])
    perm.append(list_of_lists[i][0])
    sp_500.append(list_of_lists[i][4])

#populating list for tickers
for i in range(0, 6000, 60):
    tickers.append(list_of_lists[i][2])

#create yield function that will yield each return in order
def returns_yielder(list_of_lists):
    for sublist in list_of_lists:
        yield sublist[3]

#create yielder object
gen = returns_yielder(list_of_lists)

#create list of lists with returns for each stock in each list
returns = [[next(gen) for j in range(60)] for i in range(100)]

# Convert list of lists to Pandas DataFrame with stock tickers as the column names
df = pd.DataFrame()
for i in range(100):
    new_df = pd.DataFrame({tickers[i]: returns[i]})
    df = pd.concat([df, new_df], axis=1)

#insert dates in Pandas DataFrame
df.insert(0, 'Date', dates)

#insert risk free rates in Pandas DataFrame
rf = pd.read_csv(r"C:\Users\chada\OneDrive\Desktop\econ_450\risk_free_rate.csv")
df.insert(1, 'rf_rate', rf.rf_rate)

#insert market index
df.insert(2, 'sp_500', sp_500)

print(df)

########################################################################################################################
from sklearn import linear_model

#first pass regression

#setting empty lists
coefs = []
var_resid = []
mean_r_rf = []
mean_rm_rf = []

#iterate through each column and estimate the beta, variance of residuals, and mean r-rf
for col in range(3,103):
    
    #create regression model
    lm = linear_model.LinearRegression()

    #create r - rf vector and get mean
    y = np.array(df.iloc[: , col]) - np.array(df.rf_rate)
    mean_r_rf.append(y.mean())

    #create rm - rf vector and get mean
    x = np.array(df.sp_500) - np.array(df.rf_rate)
    mean_rm_rf.append(x.mean())

    #reshape x bc sklearn is dumb
    x_reshaped = x.reshape(-1, 1)

    #estimate model
    model = lm.fit(x_reshaped , y)
    coefs.append(float(lm.coef_))

    #calculate variance of residuals and add it to a list
    resid = []
    for i in range(0,60):
        resid.append(y[i] - model.predict(x[i].reshape(-1,1)))
    resid = np.array(resid)
    var_resid.append(np.var(resid))

###################################################################################################################

#create new dataframe for second pass regression
second_pass = pd.DataFrame()
second_pass['TICK'] = tickers
second_pass['B'] = coefs
second_pass['var_resid'] = var_resid
second_pass['mean_r_rf'] = mean_r_rf
second_pass['mean_rm_rf'] = mean_rm_rf

print(second_pass)

###################################################################################################################
import statsmodels.formula.api as smf

#second pass regression
capm_test = smf.ols("mean_r_rf ~ B + var_resid", data=second_pass)
results = capm_test.fit()
print(results.summary())


# Test the null hypothesis that the beta is 0
t_test_result = results.t_test("var_resid = 0")
print(t_test_result)

# Extract the p-value from the test result
p_value = t_test_result.pvalue.item()
print(f"P-value: {p_value:.9f}")

##########################################################################################################
import seaborn as sns
import matplotlib.pyplot as plt
#graphs and figures and summary statistics


market_returns = df.iloc[0:60, 0:3]
fig, ax = plt.subplots()
ax = sns.lineplot(data = market_returns, x = 'Date', y = 'sp_500')
plt.ylabel('Return (%)')
plt.xlabel('Date')
plt.xticks(np.arange(11, 60, 12))
#ax.set_xticklabels([i for i in range(1,61)])
plt.tick_params(labelrotation=45)
plt.title('Average Market Return (S&P 500)', fontsize=15, fontweight='bold', pad=10)
plt.savefig(f'market_return', dpi=300, transparent=True)
plt.show()

print(second_pass.describe())

second_pass.to_csv('second_pass_table.csv')