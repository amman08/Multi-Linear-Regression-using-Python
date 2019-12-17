###GATHERING  DATA FROM EIKON
'''
import eikon as ek
import matplotlib.pylab as plt
import pandas as pd
from datetime import date
ek.set_app_key('53f3b6e6a2af4ee987df42330c0fcbf40f75a204')
# # Below Code is for collecting the till date data 
# tilldate= date.today()
# print(tilldate)
# tilldate=str(tilldate)
#unable to get 20 years data so divided into parts
bank = ek.get_timeseries(["JPM"], start_date = "1998-12-01", end_date ="2007-12-21", interval="daily",corax="adjusted") # JP Morgan
bank_1 = ek.get_timeseries(["JPM"], start_date = "2007-12-22", end_date ="2019-11-12", interval="daily",corax="adjusted")
#MERGED THE DATA
bank_jpm_1=pd.merge(bank,bank_1,on=['Date','HIGH', 'CLOSE', 'LOW', 'OPEN', 'COUNT', 'VOLUME'],how="outer")
bank_jpm_1.size
bank_jpm_1.to_csv("JPM.csv")
#CITY BANK
C_bank = ek.get_timeseries(["C"], start_date = "1998-12-01", end_date ="2007-12-21", interval="daily",corax="adjusted") # CITY
C_bank_1 = ek.get_timeseries(["C"], start_date = "2007-12-22", end_date ="2019-11-12", interval="daily",corax="adjusted")
CITY_BANK=pd.merge(C_bank,C_bank_1,on=['Date','HIGH', 'CLOSE', 'LOW', 'OPEN', 'COUNT', 'VOLUME'],how="outer")
CITY_BANK.size
CITY_BANK.to_csv("CITY.csv")
#Wells Forgo & Co
W_bank = ek.get_timeseries(["WFC"], start_date = "1998-12-01", end_date ="2007-12-21", interval="daily",corax="adjusted") # WELLS
W_bank_1 = ek.get_timeseries(["WFC"], start_date = "2007-12-22", end_date ="2019-11-12", interval="daily",corax="adjusted")
WELLS_BANK=pd.merge(C_bank,C_bank_1,on=['Date','HIGH', 'CLOSE', 'LOW', 'OPEN', 'COUNT', 'VOLUME'],how="outer")
WELLS_BANK.size
WELLS_BANK.to_csv("WELLS.csv")
BA_bank = ek.get_timeseries(["BAC"], start_date = "1998-12-01", end_date ="2007-12-21", interval="daily",corax="adjusted") # BOA
BA_bank_1 = ek.get_timeseries(["BAC"], start_date = "2007-12-22", end_date ="2019-11-12", interval="daily",corax="adjusted")
BOA_BANK=pd.merge(C_bank,C_bank_1,on=['Date','HIGH', 'CLOSE', 'LOW', 'OPEN', 'COUNT', 'VOLUME'],how="outer")
BOA_BANK.size
BOA_BANK.to_csv("BOA.csv")
#code for getting us_unemployement rate
US_UNEMPLOY = ek.get_timeseries(["USUNR=ECI"], start_date = "1998-12-01", end_date = "2019-11-12",interval="monthly")
US_UNEMPLOY.to_csv("unemployment.csv")
#code for getting US_inflation rate
US_INFLATION = ek.get_timeseries(["aUSWOCPIPR"], start_date = "1998-12-01", end_date = "2019-11-12",interval="yearly")
US_INFLATION.to_csv("inflation.csv")
# code for getting CONSUMER CONFIDENCE
US_CC = ek.get_timeseries(["USCONC=ECI"], start_date = "1998-12-01", end_date = "2019-11-12",interval="monthly")
US_CC.to_csv("consumer_confidence.csv")
#USA GDP
import quandl
USA_GDP = quandl.get("FRED/GDP")
USA_GDP.to_csv("gdp.csv")
# code for getting CPI
US_CPI = ek.get_timeseries(["USCPSA=ECI"], start_date = "1998-12-01", end_date = "2019-11-12",interval="monthly")
US_CPI.to_csv("cpi.csv")
# code for getting 10Y Bond yield%
Yield = ek.get_timeseries(["aUSEBM10Y"], start_date = "1998-12-01", end_date = "2019-11-12",interval="monthly")
Yield.to_csv("yield.csv")
#Gold price world wise
GOLD1 = ek.get_timeseries(["XAU="], start_date = "1998-12-01", end_date = "2007-12-21",interval="daily")
GOLD2 = ek.get_timeseries(["XAU="], start_date = "2007-12-22", end_date = "2019-11-12",interval="daily")
GOLD=pd.merge(GOLD1,GOLD2,on=['Date','HIGH', 'CLOSE', 'LOW', 'OPEN'],how="outer")
GOLD.to_csv("gold.csv")
SECTOR_INDICIES1=ek.get_timeseries([".NYUSMFNT"],start_date="1999-01-01",end_date="2007-12-21",interval="daily")
SECTOR_INDICIES2=ek.get_timeseries([".NYUSMFNT"],start_date="2007-12-22",end_date="2019-11-12",interval="daily")
'''
### CLEANING DATA AND ARRANING ALL THE DATA IN A DATA FRAME
import os
os.chdir("C:/Users/Admin/Desktop/DATASETS")
import pandas
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from matplotlib import pyplot
def bank(filename):
    bank=pandas.read_csv(filename,parse_dates=["Date"],index_col="Date")
    bank.drop(columns=['HIGH','LOW', 'OPEN', 'COUNT','VOLUME'],inplace=True)
    dt=pandas.date_range(start="1998-12-01",end="2019-11-11",freq="D") 
    idx=pandas.DatetimeIndex(dt)
    bank=bank.reindex(idx)
    bank.index.name='DATE'
    return(bank)
BOA_BANK=bank("BOA1.csv") 
BOA_BANK.rename(columns={"CLOSE":"BOA_CLOSE"},inplace=True)
CITY_BANK=bank("CITY1.csv") 
CITY_BANK.rename(columns={"CLOSE":"CITY_CLOSE"},inplace=True)
WELLS_BANK=bank("WELLS1.csv")
WELLS_BANK.rename(columns={"CLOSE":"WELLS_CLOSE"},inplace=True)
JPM_BANK=bank("JPM1.csv")
JPM_BANK.rename(columns={"CLOSE":"JPM_CLOSE"},inplace=True)

### USA GDP
usa_gdp=pandas.read_csv("gdp12.csv",parse_dates=["Date"],index_col="Date")
usa_gdp.rename(columns={"Value":"GDP_VALUE"},inplace=True)

### SETTING DATE RANGE
dt=pandas.date_range(start="1998-12-01",end="2019-11-11",freq="D") 
idx=pandas.DatetimeIndex(dt)
usa_gdp=usa_gdp.reindex(idx)
usa_gdp.interpolate(method="time",inplace=True)
### USA UNEMPLOYEMENT
usa_unemp=pandas.read_csv("unemployment.csv",parse_dates=["Date"],index_col="Date")
usa_unemp.rename(columns={"VALUE":"UNEMPLO_RATE"},inplace=True)

### USA INFLATION
usa_inf=pandas.read_csv("inflation.csv",parse_dates=["Date"],index_col="Date")
usa_inf.rename(columns={"VALUE":"INFLA_RATE"},inplace=True)

### USA CONSUMER CONFIDENCE
usa_CC=pandas.read_csv("consumer_confidence.csv",parse_dates=["Date"],index_col="Date")
usa_CC.rename(columns={"VALUE":"CC_VALUE"},inplace=True)

### USA CPI
usa_CPI=pandas.read_csv("CPI.csv",parse_dates=["Date"],index_col="Date")
usa_CPI.rename(columns={"VALUE":"CPI_VALUE"},inplace=True)

### GOLD
GOLD=pandas.read_csv("gold.csv",parse_dates=["Date"],index_col="Date")
GOLD.drop(columns=['HIGH','LOW', 'OPEN'],inplace=True)
GOLD.rename(columns={"CLOSE":"GOLD_CLOSE"},inplace=True)
### YIELD
YIELD=pandas.read_csv("yield.csv",parse_dates=["Date"],index_col="Date")
YIELD.rename(columns={"VALUE":"YIELD_VALUE"},inplace=True)

### JOINING THE MACRO DETAILS & BANK DETAILS
data=BOA_BANK.join(JPM_BANK,how="outer")
data=data.join(CITY_BANK,how="outer")
data=data.join(WELLS_BANK,how="outer")
data=data.join(usa_inf,how="outer")
data=data.join(usa_gdp,how="outer")
data=data.join(usa_unemp,how="outer")
data=data.join(usa_CC,how="outer")
data=data.join(usa_CPI,how="outer")
data=data.join(GOLD,how="outer")
data=data.join(YIELD,how="outer")

### FOR FILLING THE NAN VALUES
data.interpolate(method="time",inplace=True)
### SLICING DATA
data=data['19990101':'20191111']
data.index.name='DATE'
## CHECKING FOR NA VALUES
data.isna().sum()
data.index.name="DATE"
print(data)



###VISUALIZING THE DATA
##PAIR PLOTS
sns.pairplot(data,kind='reg')
plt.show()
### HISTOGRAMS

plt.hist(data["BOA_CLOSE"],bins=100,color='red')
plt.xlabel("BOA_CLOSE_PRICE")
plt.ylabel("FREQUENCY")
plt.title("HISTOGRAM OF BANK OF AMERICA")
plt.hist(data["CITY_CLOSE"],bins=100,color='blue')
plt.xlabel("CITY_CLOSE_PRICE")
plt.ylabel("FREQUENCY")
plt.title("HISTOGRAM OF CITY BANK")
plt.hist(data["JPM_CLOSE"],bins=100,color='indigo')
plt.xlabel("JPM_CLOSE_PRICE")
plt.ylabel("FREQUENCY")
plt.title("HISTOGRAM OF JPM BANK")
plt.hist(data["WELLS_CLOSE"],bins=100,color='navy')
plt.xlabel("WELLS_CLOSE_PRICE")
plt.ylabel("FREQUENCY")
plt.title("HISTOGRAM OF WELLS FORGO BANK")

plt.plot(data["JPM_CLOSE"],color='orange')
plt.xlabel("YEAR")
plt.ylabel("PRICE")
plt.title("JPM MORGAN BANK")

plt.plot(data["BOA_CLOSE"],color='navy')
plt.xlabel("YEAR")
plt.ylabel("PRICE")
plt.title("BANK OF AMERICA")

plt.plot(data["CITY_CLOSE"],color='black')
plt.xlabel("YEAR")
plt.ylabel("PRICE")
plt.title("CITY BANK")


plt.plot(data["WELLS_CLOSE"],color='red')
plt.xlabel("YEAR")
plt.ylabel("PRICE")
plt.title("WELLS FORGO BANK")

### CORRELATION PLOT
mask=np.zeros_like(data.corr())
traingle=np.triu_indices_from(mask)
mask[traingle]=True
plt.figure(figsize=(10,10))
sns.heatmap(data.corr(),mask=mask,annot=True,annot_kws={"size":14})
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
sns.set_style("white")

###SPLITTING DATA SETS
x=data[['CITY_CLOSE']]
y=data.drop(['BOA_CLOSE', 'JPM_CLOSE', 'CITY_CLOSE','WELLS_CLOSE'],axis=1)
x_train,x_test,y_train,y_test=train_test_split(y,x,test_size=0.2,random_state=0)
### MULTI LINEAR REGRESSION MODEL CREATION OF CITY BANK
regression=LinearRegression()
regression.fit(x_train,y_train)
print(pandas.DataFrame({"index":x_train.columns,"coeff":regression.coef_.tolist()[0]}))
pandas.DataFrame(data=regression.coef_,index=['coefficient'],columns=x_train.columns)
print("intercept:",regression.intercept_)
print('R^2 train dataset:',regression.score(x_train,y_train))
print('R^2 test dataset:',regression.score(x_test,y_test))

###PREDICTING
def predict_bankprice(dataset):
    a=regression.predict(dataset)
    predict=pandas.DataFrame.from_records(a,columns=["PREDICTED PRICE"])
    ind=pandas.DataFrame(dataset.index)
    frames=[ind,predict]
    df=pandas.concat(frames,axis=1)
    df.set_index(["DATE"],inplace=True)
    print(df)
    plt.plot(df)
    plt.legend(loc='best')
    plt.show(block=False)
    return df

###TOTAl PREDICTION
total_actual=x
ad=predict_bankprice(y)
plt.figure(figsize=(16,16))
pyplot.plot(total_actual)
pyplot.plot(ad)

###PERFORMING MODEL DIAGNOSTICS
###P_VALUES:FOR SIGNIFICANCE
##P_VALUES
x_include_constant=sm.add_constant(x_train)
model=sm.OLS(y_train,x_include_constant)
results=model.fit()
results.params
results.pvalues
print(pandas.DataFrame({"Coeffiencients":results.params,"P-Values":round(results.pvalues,2)}))
results.summary()###SUMMARY OF REGRESSION
###VARIANCE INFLATION FACTOR:CHECKING FOR MULTI COLLINEARITY
from statsmodels.stats.outliers_influence import variance_inflation_factor
VIF=[]
for i in range(0, len(x_include_constant.columns)):
        VIF.append(variance_inflation_factor(exog=x_include_constant.values,exog_idx=i))
print(VIF)
print(pandas.DataFrame({"coeffients":x_include_constant.columns,"VIF":np.around(VIF,3)}))

###RESIDUALS

###ANALYSING RESIDUALS STATS MODEL
x_include_constant=sm.add_constant(x_train)
model=sm.OLS(y_train,x_include_constant)
results=model.fit()

### CODE FOR RESIDUALS
results.resid

###CORELLATION BETWEEN y_train AND PREDICTED y_train
act_pred=pandas.DataFrame({"actual":y_train["CITY_CLOSE"],"predicted":results.fittedvalues})
corr=round(act_pred["actual"].corr(act_pred["predicted"]),2)
print(corr)
##GRAPH OF ACTUAL VS PREDICTION
###SCATTER PLOT
plt.figure(figsize=(7,7))
plt.scatter(x=act_pred["actual"],y=act_pred["predicted"],color="red",alpha=0.6)
plt.plot(act_pred["actual"],act_pred["actual"],color="black")
plt.xlabel("ACTUAL CLOSE VALUES",fontsize=12)
plt.ylabel("PREDICTED CLOSE VALUES",fontsize=12)
plt.title(f'ACTUAL  VS PREDICTED VALUES (corr{corr})',fontsize=18 )
plt.show()
##RESIDUAL VS PREDICTED VALUES
plt.figure(figsize=(7,7))
plt.scatter(x=results.fittedvalues,y=results.resid,color="navy",alpha=0.6)

plt.xlabel("predicted values",fontsize=12)
plt.ylabel("residuals",fontsize=12)
plt.title("RESIDUALS VS FITTED VALUES",fontsize=18 )
plt.show()

####DISTRIBUTION OF RESIDUALS
residual_mean=round(results.resid.mean(),3)
residual_skew=round(results.resid.skew(),3)
plt.figure(figsize=(10,10))
sns.distplot(results.resid,color="blue")
plt.title("RESIDUAL DISTRIBUTION")
 ####MEAN SQUARED ERROR
results.mse_resid
results.rsquared
###PREDICTING AND RANGE
##ROOT MEAN SQUARE ERROR
RMSE=np.sqrt(results.mse_resid)

pandas.DataFrame({"R-Squared":[results.rsquared],"Mean Square Error":[results.mse_resid],"Root Mean Square":np.sqrt(results.mse_resid)},index=["JPM_CLOSE"])

print("one standard deviation:",np.sqrt(results.mse_resid))

def predict_Bank_price(dataset):
    a=regression.predict(dataset)
    predict=pandas.DataFrame.from_records(a,columns=["PREDICTED PRICE"])
    ind=pandas.DataFrame(dataset.index)
    frames=[ind,predict]
    df=pandas.concat(frames,axis=1)
    df.set_index(["DATE"],inplace=True)
    print(df)
    plt.plot(df)
    plt.legend(loc='best')
    plt.show(block=False)
    return df

b=predict_Bank_price(y)
plt.plot(b)
plt.plot(x)
###LINEAR REGRESSION BOA_CLOSE

print("\n\n\n LINEAR REGRESSION BOA_CLOSE")
x1=data[['BOA_CLOSE']]
y1=data.drop(['BOA_CLOSE', 'JPM_CLOSE', 'CITY_CLOSE','WELLS_CLOSE','CPI_VALUE','GOLD_CLOSE','YIELD_VALUE','CC_VALUE'],axis=1)
x_train1,x_test1,y_train1,y_test1=train_test_split(y1,x1,test_size=0.2,random_state=0)
regression1=LinearRegression()
regression1.fit(x_train1,y_train1)
print(pandas.DataFrame({"index":x_train1.columns,"coeff":regression1.coef_.tolist()[0]}))
pandas.DataFrame(data=regression1.coef_,index=['coefficient'],columns=x_train1.columns)
print("intercept:",regression1.intercept_)
print('R^2 train dataset:',regression1.score(x_train1,y_train1))
print('R^2 test dataset:',regression1.score(x_test1,y_test1))

###PERFORMING MODEL DIAGNOSTICS
###P_VALUES:FOR SIGNIFICANCE
##P_VALUES
x1_include_constant=sm.add_constant(x_train1)
model=sm.OLS(y_train1,x1_include_constant)
results1=model.fit()
results1.params
results1.pvalues
print(pandas.DataFrame({"Coeffiencients":results1.params,"P-Values":round(results1.pvalues,2)}))
results1.summary()###SUMMARY OF REGRESSION
###VARIANCE INFLATION FACTOR:CHECKING FOR MULTI COLLINEARITY
from statsmodels.stats.outliers_influence import variance_inflation_factor
VIF1=[]
for i in range(0, len(x1_include_constant.columns)):
        VIF1.append(variance_inflation_factor(exog=x1_include_constant.values,exog_idx=i))
print(VIF1)
print(pandas.DataFrame({"coeffients":x1_include_constant.columns,"VIF":np.around(VIF1,3)}))

###RESIDUALS

###ANALYSING RESIDUALS STATS MODEL
x1_include_constant=sm.add_constant(x_train1)
model1=sm.OLS(y_train1,x1_include_constant)
results1=model.fit()

### CODE FOR RESIDUALS
results1.resid

###CORELLATION BETWEEN y_train AND PREDICTED y_train
act_pred1=pandas.DataFrame({"actual":y_train1['BOA_CLOSE'],"predicted":results1.fittedvalues})
corr1=round(act_pred1["actual"].corr(act_pred1["predicted"]),2)
print(corr1)
##GRAPH OF ACTUAL VS PREDICTION
###SCATTER PLOT
plt.figure(figsize=(7,7))
plt.scatter(x=act_pred1["actual"],y=act_pred1["predicted"],color="red",alpha=0.6)
plt.plot(act_pred1["actual"],act_pred1["actual"],color="black")
plt.xlabel("ACTUAL CLOSE VALUES",fontsize=12)
plt.ylabel("PREDICTED CLOSE VALUES",fontsize=12)
plt.title(f'ACTUAL  VS PREDICTED VALUES (corr{corr})',fontsize=18 )
plt.show()
##RESIDUAL VS PREDICTED VALUES
plt.figure(figsize=(7,7))
plt.scatter(x=results1.fittedvalues,y=results1.resid,color="navy",alpha=0.6)

plt.xlabel("predicted values",fontsize=12)
plt.ylabel("residuals",fontsize=12)
plt.title("RESIDUALS VS FITTED VALUES",fontsize=18 )
plt.show()

####DISTRIBUTION OF RESIDUALS
residual_mean1=round(results1.resid.mean(),3)
residual_skew1=round(results1.resid.skew(),3)
plt.figure(figsize=(10,10))
sns.distplot(results1.resid,color="blue")
plt.title("RESIDUAL DISTRIBUTION")
 ####MEAN SQUARED ERROR
results1.mse_resid
results1.rsquared
###PREDICTING AND RANGE
##ROOT MEAN SQUARE ERROR
RMSE1=np.sqrt(results1.mse_resid)

pandas.DataFrame({"R-Squared":[results1.rsquared],"Mean Square Error":[results1.mse_resid],"Root Mean Square":np.sqrt(results1.mse_resid)},index=["JPM_CLOSE"])

print("one standard deviation:",np.sqrt(results1.mse_resid))

def predict_Bank_price1(dataset):
    a=regression1.predict(dataset)
    predict=pandas.DataFrame.from_records(a,columns=["PREDICTED PRICE"])
    ind=pandas.DataFrame(dataset.index)
    frames=[ind,predict]
    df=pandas.concat(frames,axis=1)
    df.set_index(["DATE"],inplace=True)
    print(df)
    plt.plot(df)
    plt.legend(loc='best')
    plt.show(block=False)
    return df

b1=predict_Bank_price1(y1)
plt.plot(b1)
plt.plot(x1)

####LINEAR REGRESSION JPM_CLOSE#################################

x2=data[['JPM_CLOSE']]
y2=data.drop(['BOA_CLOSE', 'JPM_CLOSE', 'CITY_CLOSE','WELLS_CLOSE','CPI_VALUE','GOLD_CLOSE','YIELD_VALUE','INFLA_RATE'],axis=1)
x_train2,x_test2,y_train2,y_test2=train_test_split(y2,x2,test_size=0.2,random_state=0)
### MULTI LINEAR REGRESSION MODEL CREATION OF CITY BANK
regression2=LinearRegression()
regression2.fit(x_train2,y_train2)
print(pandas.DataFrame({"index":x_train2.columns,"coeff":regression2.coef_.tolist()[0]}))
pandas.DataFrame(data=regression2.coef_,index=['coefficient'],columns=x_train2.columns)
print("intercept:",regression2.intercept_)
print('R^2 train dataset:',regression2.score(x_train2,y_train2))
print('R^2 test dataset:',regression2.score(x_test2,y_test2))

###PERFORMING MODEL DIAGNOSTICS########################################
###P_VALUES:FOR SIGNIFICANCE
##P_VALUES
x2_include_constant=sm.add_constant(x_train2)
model2=sm.OLS(y_train2,x2_include_constant)
results2=model2.fit()
results2.params
results2.pvalues
print(pandas.DataFrame({"Coeffiencients":results2.params,"P-Values":round(results2.pvalues,2)}))
results.summary()###SUMMARY OF REGRESSION
###VARIANCE INFLATION FACTOR:CHECKING FOR MULTI COLLINEARITY
from statsmodels.stats.outliers_influence import variance_inflation_factor
VIF2=[]
for i in range(0, len(x2_include_constant.columns)):
        VIF2.append(variance_inflation_factor(exog=x2_include_constant.values,exog_idx=i))
print(VIF2)
print(pandas.DataFrame({"coeffients":x2_include_constant.columns,"VIF2":np.around(VIF2,3)}))

###RESIDUALS

###ANALYSING RESIDUALS STATS MODEL
x2_include_constant=sm.add_constant(x_train2)
model2=sm.OLS(y_train2,x2_include_constant)
results2=model2.fit()

### CODE FOR RESIDUALS
results2.resid

###CORELLATION BETWEEN y_train AND PREDICTED y_train
act_pred2=pandas.DataFrame({"actual":y_train2["JPM_CLOSE"],"predicted":results2.fittedvalues})
corr2=round(act_pred2["actual"].corr(act_pred2["predicted"]),2)
print(corr2)
##GRAPH OF ACTUAL VS PREDICTION
###SCATTER PLOT
plt.figure(figsize=(7,7))
plt.scatter(x=act_pred2["actual"],y=act_pred2["predicted"],color="red",alpha=0.6)
plt.plot(act_pred2["actual"],act_pred2["actual"],color="black")
plt.xlabel("ACTUAL CLOSE VALUES",fontsize=12)
plt.ylabel("PREDICTED CLOSE VALUES",fontsize=12)
plt.title(f'ACTUAL  VS PREDICTED VALUES (corr{corr})',fontsize=18 )
plt.show()
##RESIDUAL VS PREDICTED VALUES
plt.figure(figsize=(7,7))
plt.scatter(x=results2.fittedvalues,y=results2.resid,color="navy",alpha=0.6)

plt.xlabel("predicted values",fontsize=12)
plt.ylabel("residuals",fontsize=12)
plt.title("RESIDUALS VS FITTED VALUES",fontsize=18 )
plt.show()

####DISTRIBUTION OF RESIDUALS
residual_mean2=round(results2.resid.mean(),3)
residual_skew2=round(results2.resid.skew(),3)
plt.figure(figsize=(10,10))
sns.distplot(results2.resid,color="blue")
plt.title("RESIDUAL DISTRIBUTION")
 ####MEAN SQUARED ERROR
print(results2.mse_resid)
print(results2.rsquared)
###PREDICTING AND RANGE
##ROOT MEAN SQUARE ERROR
RMSE2=np.sqrt(results2.mse_resid)

pandas.DataFrame({"R-Squared":[results2.rsquared],"Mean Square Error":[results2.mse_resid],"Root Mean Square":np.sqrt(results2.mse_resid)},index=["JPM_CLOSE"])

print("one standard deviation:",np.sqrt(results2.mse_resid)*1)
print("Two standard deviation:",np.sqrt(results2.mse_resid)*2)
print("Three standard deviation:",np.sqrt(results2.mse_resid)*3)
def predict_Bank_price(dataset):
    a=regression2.predict(dataset)
    upper=a+1*RMSE2
    lower=a-1*RMSE2
    predict=pandas.DataFrame.from_records(a,columns=["PREDICTED PRICE"])
    upper=pandas.DataFrame.from_records(upper,columns=["UPPER PRICE"])
    lower=pandas.DataFrame.from_records(lower,columns=["LOWER PRICE"])
    ind=pandas.DataFrame(dataset.index)
    frames=[ind,predict,upper,lower]
    df=pandas.concat(frames,axis=1)
    df.set_index(["DATE"],inplace=True)
    print(df)
    plt.plot(df)
    plt.legend(loc='best')
    plt.show(block=False)
    return df

b2=predict_Bank_price(y2)
plt.figure(figsize=(20,10))
plt.plot(b2)
plt.plot(x2)
plt.legend(loc='best')
plt.title("JPM_BANK")
plt.show(block=False)














##################################################







def predict_Bank_price2(dataset):
    a=regression2.predict(dataset)
    predict=pandas.DataFrame.from_records(a,columns=["PREDICTED PRICE"])
    ind=pandas.DataFrame(dataset.index)
    frames=[ind,predict]
    df=pandas.concat(frames,axis=1)
    df.set_index(["DATE"],inplace=True)
    print(df)
    plt.plot(df)
    plt.legend(loc='best')
    plt.show(block=False)
    return df

b2=predict_Bank_price2(y2)
plt.plot(b2)
plt.plot(x2)
plt.title("JPM_BANK")


####LINEAR REGRESSION WELLS_CLOSED
x3=data[['WELLS_CLOSE']]
y3=data.drop(['BOA_CLOSE', 'JPM_CLOSE', 'CITY_CLOSE','WELLS_CLOSE','CPI_VALUE','GOLD_CLOSE','YIELD_VALUE','UNEMPLO_RATE'],axis=1)
x_train3,x_test3,y_train3,y_test3=train_test_split(y3,x3,test_size=0.2,random_state=0)
### MULTI LINEAR REGRESSION MODEL CREATION OF CITY BANK
regression3=LinearRegression()
regression3.fit(x_train3,y_train3)
print(pandas.DataFrame({"index":x_train3.columns,"coeff":regression3.coef_.tolist()[0]}))
pandas.DataFrame(data=regression3.coef_,index=['coefficient'],columns=x_train3.columns)
print("intercept:",regression3.intercept_)
print('R^2 train dataset:',regression3.score(x_train3,y_train3))
print('R^2 test dataset:',regression3.score(x_test3,y_test3))

###PERFORMING MODEL DIAGNOSTICS
###P_VALUES:FOR SIGNIFICANCE
##P_VALUES
x3_include_constant=sm.add_constant(x_train3)
model3=sm.OLS(y_train3,x3_include_constant)
results3=model3.fit()
results3.params
results3.pvalues
print(pandas.DataFrame({"Coeffiencients":results3.params,"P-Values":round(results3.pvalues,2)}))
results3.summary()###SUMMARY OF REGRESSION
###VARIANCE INFLATION FACTOR:CHECKING FOR MULTI COLLINEARITY
from statsmodels.stats.outliers_influence import variance_inflation_factor
VIF3=[]
for i in range(0, len(x3_include_constant.columns)):
        VIF3.append(variance_inflation_factor(exog=x3_include_constant.values,exog_idx=i))
print(VIF3)
print(pandas.DataFrame({"coeffients":x3_include_constant.columns,"VIF3":np.around(VIF3,3)}))

###RESIDUALS

###ANALYSING RESIDUALS STATS MODEL
x3_include_constant=sm.add_constant(x_train3)
model3=sm.OLS(y_train3,x3_include_constant)
results3=model3.fit()

### CODE FOR RESIDUALS
results3.resid

###CORELLATION BETWEEN y_train AND PREDICTED y_train
act_pred3=pandas.DataFrame({"actual":y_train3["WELLS_CLOSE"],"predicted":results3.fittedvalues})
corr3=round(act_pred3["actual"].corr(act_pred3["predicted"]),2)
print(corr3)
##GRAPH OF ACTUAL VS PREDICTION
###SCATTER PLOT
plt.figure(figsize=(7,7))
plt.scatter(x=act_pred3["actual"],y=act_pred3["predicted"],color="red",alpha=0.6)
plt.plot(act_pred3["actual"],act_pred3["actual"],color="black")
plt.xlabel("ACTUAL CLOSE VALUES",fontsize=12)
plt.ylabel("PREDICTED CLOSE VALUES",fontsize=12)
plt.title(f'ACTUAL  VS PREDICTED VALUES (corr{corr})',fontsize=18 )
plt.show()
##RESIDUAL VS PREDICTED VALUES
plt.figure(figsize=(7,7))
plt.scatter(x=results3.fittedvalues,y=results3.resid,color="navy",alpha=0.6)

plt.xlabel("predicted values",fontsize=12)
plt.ylabel("residuals",fontsize=12)
plt.title("RESIDUALS VS FITTED VALUES",fontsize=18 )
plt.show()

####DISTRIBUTION OF RESIDUALS
residual_mean3=round(results3.resid.mean(),3)
residual_skew3=round(results3.resid.skew(),3)
plt.figure(figsize=(10,10))
sns.distplot(results3.resid,color="blue")
plt.title("RESIDUAL DISTRIBUTION")
 ####MEAN SQUARED ERROR
results3.mse_resid
results3.rsquared
###PREDICTING AND RANGE
##ROOT MEAN SQUARE ERROR
RMSE3=np.sqrt(results3.mse_resid)

pandas.DataFrame({"R-Squared":[results3.rsquared],"Mean Square Error":[results3.mse_resid],"Root Mean Square":np.sqrt(results3.mse_resid)},index=["WELLS_CLOSE"])

print("one standard deviation:",np.sqrt(results3.mse_resid))

def predict_Bank_price3(dataset):
    a=regression3.predict(dataset)
    upper=a+1*RMSE3
    lower=a-1*RMSE3
    predict=pandas.DataFrame.from_records(a,columns=["PREDICTED PRICE"])
    upper=pandas.DataFrame.from_records(upper,columns=["UPPER PRICE"])
    lower=pandas.DataFrame.from_records(lower,columns=["LOWER PRICE"])
    ind=pandas.DataFrame(dataset.index)
    frames=[ind,predict,upper,lower]
    df=pandas.concat(frames,axis=1)
    df.set_index(["DATE"],inplace=True)
    print(df)
    plt.plot(df)
    plt.legend(loc='best')
    plt.show(block=False)
    return df

b3=predict_Bank_price3(y3)
plt.plot(b3)
plt.plot(x3)
plt.title("WELLS_BANK")

b3["actual"]=x3























