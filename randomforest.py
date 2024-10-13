pip install yfinance
import yfinance as yf
import pandas as pd
stock = yf.Ticker("TCS.NS")
stock=stock.history(period='max')
del stock['Dividends']
del stock['Stock Splits']
stock['Tomorrow open']= stock['Close'].shift(-1)
stock['act tom open']=stock['Open'].shift(-1)
stock['target']=(stock['Tomorrow open']>stock['Close']).astype(int)
sp=stock.loc['2003-01-01':].copy()
sp
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.metrics import precision_score as PS
model=RFC(n_estimators=100,min_samples_split=100,random_state=1)
train=sp.iloc[:-100]
test=sp.iloc[-100:]
preds=model.predict(test[predicts])
preds=pd.Series(preds,index=test.index)
precision=PS(test['target'],preds)
combined=pd.concat([test['target'],preds],axis=1)
combined.plot()
def predict(train,test,predicts,model):
    model.fit(train[predicts],train['target'])
    preds=model.predict(test[predicts])
    preds=pd.Series(preds,index=test.index,name='predictions')
    combined=pd.concat([test['target'],preds],axis=1)
    return combined
def backtest(data,model,predicts,start=2500,step=250):
    all_predictions=[]
    for i in range(start,data.shape[0],step):
        train=data.iloc[:i].copy()
        test=data.iloc[i:(i+step)].copy()
        predictions=predict(train,test,predicts,model)
        all_predictions.append(predictions)
    return pd.concat(all_predictions)
predictions=backtest(sp,model,predicts)
predictions['predictions'].value_counts()
PS(predictions['target'],predictions['predictions'])
predictions['target'].value_counts()/predictions.shape[0]
horizons=[2,5,60,250,1000]
new_pedictors=[]
for horizon in horizons:
    rolling_avgs=sp.rolling(horizon).mean()
    ratio_column=f"close_ratio_{horizon}"
    sp[ratio_column]=sp['Close']/rolling_avgs['Close']
    trend_column=f'Trend_{horizon}'
    sp[trend_column]=sp.shift(1).rolling(horizon).sum()['target']
    new_pedictors+=[ratio_column,trend_column]

sp=sp.dropna()
sp
new_pedictors
model=RFC(n_estimators=200,min_samples_split=50,random_state=1)
def predict(train,test,predicts,model):
    model.fit(train[predicts],train['target'])
    preds=model.predict_proba(test[predicts])[:,1]
    preds[preds>=0.6]=1
    preds[preds<.6]=0
    preds=pd.Series(preds,index=test.index,name='predictions')
    combined=pd.concat([test['target'],preds],axis=1)
    return combined
predictions=backtest(sp,model,new_pedictors)
predictions['predictions'].value_counts()
PS(predictions['target'],predictions['predictions'])
combined=pd.concat([predictions['target'],predictions['predictions']],axis=1)
combined.plot()












