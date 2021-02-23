import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def plotTimeSeries(timeSeries,styles=None,colors=None):
    if styles==None: styles=['-' for i in range(0,len(timeSeries.columns))] 
    if colors==None: colors=['darkblue']+['C'+str(i) for i in range(1,len(timeSeries.columns))]
    fig, ax = plt.subplots()
    for col, style, color in zip(timeSeries.columns,styles,colors):
        timeSeries[col].plot(style=style,ax=ax,color=color,label=col,figsize=(15,7))
        plt.legend(loc='upper left')
    plt.show()

def goodnessOfFitStats(timeSeries,observation,estimated):
    standarErrors = ((timeSeries[observation]-timeSeries[estimated])**2).to_frame()
    standarErrors.columns = [estimated+'_MSE']
    standarErrors[estimated+'_MAE'] = abs(timeSeries[observation]-timeSeries[estimated])
    standarErrors[estimated+'_MAPE'] = abs(timeSeries[observation]-timeSeries[estimated])\
                                      /timeSeries[observation]
    return standarErrors.dropna().mean()
    
def simpleExpSmoothing(timeSeries,observation,alpha=0.5):
    expSmoothing = timeSeries[[observation]]
    S1 =[timeSeries.iloc[0][observation]]
    for t in range (1,len(timeSeries)):
        S1.append(alpha*timeSeries.iloc[t][observation]+(1-alpha)*S1[t-1])
    expSmoothing[observation+'_est'] = S1
    return expSmoothing

def BrownsSmoothing(timeSeries,observation,alpha=0.5):
    brownsSmooothing = timeSeries[[observation]]
    S1 =[timeSeries.iloc[0][observation]]
    S2 =[timeSeries.iloc[0][observation]]
    for t in range (1,len(timeSeries)):
        S1.append(alpha*timeSeries.iloc[t][observation]+(1-alpha)*S1[t-1])
    brownsSmooothing['S\''] = S1
    for t in range (1,len(timeSeries)):
        S2.append(alpha*S1[t]+(1-alpha)*S2[t-1])
    brownsSmooothing['S\'\''] = S2
    brownsSmooothing['a'] = 2*brownsSmooothing['S\'']-brownsSmooothing['S\'\''] 
    brownsSmooothing['b'] = ((alpha)/(1-alpha))*(brownsSmooothing['S\'']-brownsSmooothing['S\'\''])
    brownsSmooothing[observation+'_est'] = brownsSmooothing['a']+brownsSmooothing['b']
    return brownsSmooothing

def WintersSmoothing(timeSeries,observation,s,alpha=0.5,beta=0.5,gamma=0.5):
    wintersSmooothing = timeSeries[[observation]]
    L0 = wintersSmooothing[:s].mean()[0]
    T0 = wintersSmooothing.iloc[len(timeSeries)-1][observation]/\
         wintersSmooothing.iloc[s][observation]
    S = [wintersSmooothing.iloc[t][observation]/L0 for t in range(0,s)]
    L = [None]*s+[L0]
    T = [None]*s+[T0]
    S = S+[gamma*(wintersSmooothing.iloc[s][observation]/L[s])+(1-gamma)*S[0]]
    XtEst = [None]*s+[(L[s]+T[s])*S[0]]
    for t in range(s+1,len(timeSeries)):
        Xt = wintersSmooothing.iloc[t][observation]
        L.append(alpha*(Xt/S[t-s])+(1-alpha)*(L[t-1]+T[t-1]))
        T.append(beta*(L[t]-L[t-1])+(1-beta)*T[t-1])
        S.append(gamma*(Xt/L[t])+(1-gamma)*S[t-s])
        XtEst.append((L[t]+T[t])*S[t-s])
    wintersSmooothing['L'] = L
    wintersSmooothing['T'] = T
    wintersSmooothing['S'] = S
    wintersSmooothing[observation+'_est'] = XtEst
    return wintersSmooothing

def AditiveDecomposition(timeSeries,Xt,s):
    x = 'Index'
    timeSeries[x] = [i+1 for i in range(0,len(timeSeries))]
    timeSeries['Season'] = [(i+1)%s for i in range(0,len(timeSeries))]
    timeSeries[['Season',Xt]].groupby('Season').mean()
    X = timeSeries[x]
    X = sm.add_constant(X)
    linearRegressionModel = sm.OLS(timeSeries[Xt],X).fit()
    timeSeries['Trend'] = linearRegressionModel.predict()
    timeSeries['Detrended'+Xt] = timeSeries[Xt]-timeSeries['Trend']
    seasonalityCoefPD = timeSeries[['Season','Detrended'+Xt]].groupby('Season').mean()
    seasonalityCoefPD = seasonalityCoefPD.rename(columns={'Detrended'+Xt:'Seasonal'})
    timeSeries = timeSeries.join(seasonalityCoefPD,on='Season')
    timeSeries[Xt+'_est'] = timeSeries['Trend']+timeSeries['Seasonal']
    timeSeries['Noise'] = timeSeries[Xt]-timeSeries[Xt+'_est']
    return timeSeries

def plot_ar_p_roots(ar_p_model,figsize=(10,10)):
    theta = np.linspace(0, 2*np.pi, 100)
    X = [z.real for z in ar_p_model.arroots] 
    Y = [z.imag for z in ar_p_model.arroots] 
    limit = max(abs(max(X)),abs(min(X)),abs(max(Y)),abs(min(Y)))+0.5
    plt.figure(figsize=figsize)
    plt.xlim(-1*limit,limit)
    plt.ylim(-1*limit,limit)
    plt.plot(np.cos(theta),np.sin(theta),c='blue')
    plt.scatter(X,Y)
    
def plot_ma_q_roots(ma_q_model,figsize=(10,10)):
    theta = np.linspace(0, 2*np.pi, 100)
    X = [z.real for z in ma_q_model.maroots] 
    Y = [z.imag for z in ma_q_model.maroots] 
    limit = max(abs(max(X)),abs(min(X)),abs(max(Y)),abs(min(Y)))+0.5
    plt.figure(figsize=figsize)
    plt.xlim(-1*limit,limit)
    plt.ylim(-1*limit,limit)
    plt.plot(np.cos(theta),np.sin(theta),c='blue')
    plt.scatter(X,Y)