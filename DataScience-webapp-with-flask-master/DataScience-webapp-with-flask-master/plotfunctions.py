# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 13:29:55 2018

@author: Alvaro
"""
import code
from logging.handlers import DEFAULT_SOAP_LOGGING_PORT
import time
from turtle import title
from matplotlib import legend
from matplotlib.axis import YAxis
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import plotly
import pandas as pd
import matplotlib.pyplot as mp
import plotly.express as px
from plotly import tools
import os
import json
from plotly.subplots import make_subplots
from plotly.offline import iplot
from matplotlib.ticker import PercentFormatter
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import plotly.offline as py
import plotly.graph_objs as go
import plotly.figure_factory as ff
import plotly.tools as tls
from plotly.subplots import make_subplots
from plotly.offline import plot, iplot, init_notebook_mode
init_notebook_mode(connected=False)


#Plot ROC curves
def plot_ROC(X, y, mod, cv):
    from sklearn.metrics import roc_curve, auc
    from sklearn.model_selection import StratifiedKFold
    
    cv = StratifiedKFold(n_splits=cv)

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    i = 0
    for train, Xpred in cv.split(X,y):
        probas_ = mod.fit(X[train], y[train]).predict(X[Xpred])
        # Compute ROC curve and area the curve
        fpr, tpr, mean_fpr= roc_curve(y[Xpred], probas_[:, 1])
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3,
                 label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

        i += 1
    #figure = plt.figure()
    plt.gcf().clear()
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Luck', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc="lower right")
    from io import BytesIO
    figfile = BytesIO()
    plt.savefig(figfile, format='png')
    figfile.seek(0)  # rewind to beginning of file
    import base64
    figdata_png = base64.b64encode(figfile.getvalue())
    return figdata_png
    
def plot_predVSreal(X, y, mod, cv):
    from sklearn.model_selection import cross_val_predict
    # cross_val_predict returns an array of the same size as `y` where each entry
    # is a prediction obtained by cross validation:
    predicted = cross_val_predict(mod, X, y, cv=cv)
    plt.gcf().clear()
    plt.scatter(y, predicted, edgecolors=(0, 0, 0))
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
    plt.xlabel('Measured')
    plt.ylabel('Predicted')
    from io import BytesIO
    figfile = BytesIO()
    plt.savefig(figfile, format='png')
    figfile.seek(0)  # rewind to beginning of file
    import base64
    figdata_png = base64.b64encode(figfile.getvalue())
    return figdata_png

def plot_histsmooth(ds, histogram):
    sns.set()
    plt.gcf().clear()
    for col in histogram:
        sns.distplot(ds[col], label = col)
    from io import BytesIO
    plt.xlabel('')
    plt.legend()
    figfile = BytesIO()
    plt.savefig(figfile, format='png')
    figfile.seek(0)  # rewind to beginning of file
    import base64
    figdata_png = base64.b64encode(figfile.getvalue())
    return figdata_png

def plot_correlations(ds, corr, corrcat):
    sns.set()
    plt.gcf().clear()
    if corrcat != '': sns.pairplot(ds[corr], hue = corrcat)
    else: sns.pairplot(ds[corr])
    from io import BytesIO
    figfile = BytesIO()
    plt.savefig(figfile, format='png')
    figfile.seek(0)  # rewind to beginning of file
    import base64
    figdata_png = base64.b64encode(figfile.getvalue())
    return figdata_png

def plot_boxplot(ds, boxcat, boxconcat):
    sns.set()
    plt.gcf().clear()
    with sns.axes_style(style='ticks'):
        sns.boxplot(boxcat, dataset=ds, kind="box")
        sns.boxplot(boxconcat, dataset=ds, kind="box")
    from io import BytesIO
    plt.xlabel(boxcat)
    plt.ylabel(boxconcat)
    figfile = BytesIO()
    plt.savefig(figfile, format='png')
    figfile.seek(0)  # rewind to beginning of file
    import base64
    figdata_png = base64.b64encode(figfile.getvalue())
    return figdata_png

def itachi(ds):   
    sauske = {}
    temp = ds.astype({'Group code':'string'})['Group code'].str.split(pat ="\\s*",expand=True).add_prefix('GC')
    ds['GC1'] = pd.Series(temp['GC1'])
    ds['GC2'] = pd.Series(temp['GC2'])
    #print(ds['GC1'].unique())
    #print(ds['GC2'].unique())
    #print(ds['Group code text'].unique())
    ds = ds.dropna(axis=0, subset=['Duration'])
    ds = ds.dropna(axis=0, subset=['Group code text'])

    #ds['GCText1'] = ds.apply(lambda row: row['Group code text'].split(" ", 1)[0].strip().upper(), axis=1)
    #ds['GCText2'] = ds.apply(lambda row: row['Group code text'].split(" ", 1)[len(row['Group code text'].split(" ", 1))-1].strip().upper(), axis=1)
    # df1 = df1[df1['GCText1'] != 'PLANNED']

    def calculateMin(x):
        arr = x.split('.')
        if arr[0] != '' and arr[1] != '':
            return int(arr[0])*60 + int(arr[1])
        else:
            return 0
    ds['DurInMin'] = ds.apply(lambda row: calculateMin(row['Duration']), axis=1)
    #print(ds)
    #print(ds.groupby(['GC1', 'GCText1']).size())
    #print(ds.groupby(['GC2', 'GCText2']).size())
    #print(ds.groupby(['Cause code', 'Cause code text']).size())
    
    # plot the graph
    tmp1= ds.groupby(['GC1', 'Group code text'])['DurInMin'].sum().nlargest(10).reset_index(name ='DurSum')
    tmp2= ds.groupby(['GC1', 'Group code text']).size().nlargest(10).reset_index(name ='FreqCount')
    df2 = tmp1.merge(tmp2,on=['GC1', 'Group code text'],how="inner")
    
    #df2 = ds.groupby(['GCText1','Cause code'])['DurInMin'].sum().reset_index(name ='DurInMin')
    #df2.plot(x='GCText1', y=['DurInMin'], kind="line", figsize=(9, 8))
    #sauske['GC1'] = '/static/figure1.png'
    #mp.savefig(os.path.join('static', "figure1.png"))  # mp.show()
    
    title = 'Stoppage Duration Graph'
    
    trace1 = go.Bar(
    x=df2['Group code text'],
    y=df2['DurSum'],
    name='No.of.StoppageDurations',
    marker=dict(color='rgb(34,163,192)'))

    trace2 = go.Scatter(
    x=df2['Group code text'],
    y=df2['FreqCount'],
    name='FrequencyCount',
    yaxis='y2')
        
    
    
    data = [trace1,trace2]

    chart1 = dict(data=data)
    chart1 = make_subplots(specs=[[{"secondary_y": True}]])
    chart1.add_trace(trace1)
    chart1.add_trace(trace2,secondary_y=True)
    iplot(chart1)
    #chart1 = px.bar(df2,
                   #x='GC1',
                   #y='FreqCount',
                   #color='GCText1')
    chart1.update_layout(xaxis={'categoryorder': 'total descending'})
    graphJSON1 = json.dumps(chart1, cls=plotly.utils.PlotlyJSONEncoder)
    sauske['GC1'] = graphJSON1

    
   
    
    # plot the graph
    df2 = ds.groupby(['Group code text','Cause code'])['DurInMin'].sum().nlargest(10).reset_index(name ='DurSum')
    #df2.plot(x='GCText1', y=['Cause code'], kind="line", figsize=(9, 8))
    #sauske['GC2'] = '/static/figure2.png'
    #mp.savefig(os.path.join('static', "figure2.png"))  # mp.show()
    chart2 = px.bar(df2,
                   x='Group code text',
                   y='Cause code',
                   color_discrete_sequence = ['#1E6C8D ']*len(df2),
                   text_auto=True)
    chart2.update_layout(xaxis={'categoryorder': 'total descending'},showlegend=True)
    graphJSON2 = json.dumps(chart2, cls=plotly.utils.PlotlyJSONEncoder)
    sauske['GC2'] = graphJSON2
    
    # grouping
    df3 = ds.groupby(['Cause code text'])['DurInMin'].sum().nlargest(10).reset_index(name ='DurInMin')
    # plot the dataframe
    #df3.plot(x='GCText1', y=['DurInMin'], kind="bar", figsize=(9, 8))
    #sauske['Stoppage occurs in GCText1'] = '/static/figure3.png'
   #mp.savefig(os.path.join('static', "figure3.png"))  # mp.show()
    chart3 = px.bar(df3,
                   x='Cause code text',
                   y='DurInMin',
                   color_discrete_sequence = ['#B33996 ']*len(df3),
                   template= 'plotly_white',text_auto=True)
    chart3.update_layout(barmode='stack', xaxis={'categoryorder': 'total descending'})
    graphJSON3 = json.dumps(chart3, cls=plotly.utils.PlotlyJSONEncoder)
    sauske['GC3'] = graphJSON3
    
    # grouping
    df4 = ds.groupby(['Cause code'])['DurInMin'].sum().nlargest(10).reset_index(name ='DurSum')
    # plot the dataframe
    #df4.plot(x='GCText2', y=['DurInMin'], kind="bar", figsize=(9, 8))
    #sauske['Stoppage occurs in GCText2'] = '/static/figure4.png'
    #mp.savefig(os.path.join('static', "figure4.png"))  # mp.show()
    chart4 = px.bar(df4,
                   x='Cause code',
                   y='DurSum',
                   color_discrete_sequence = ['#39B362']*len(df4),
                   template= 'plotly_white',
                   text_auto=True)
    chart4.update_layout(barmode='stack', xaxis={'categoryorder': 'total descending'})
    graphJSON4 = json.dumps(chart4, cls=plotly.utils.PlotlyJSONEncoder)
    sauske['GC4'] = graphJSON4

    df5 = ds.groupby(['Group code'])['DurInMin'].sum().nlargest(10).reset_index(name ='DurSum')
    # plot the dataframe
    #df4.plot(x='GCText2', y=['DurInMin'], kind="bar", figsize=(9, 8))
    #sauske['Stoppage occurs in GCText2'] = '/static/figure4.png'
    #mp.savefig(os.path.join('static', "figure4.png"))  # mp.show()
    chart5 = px.bar(df5,
                   x='Group code',
                   y='DurSum',
                   color_discrete_sequence = ['#39B362']*len(df5),
                   template= 'plotly_white',
                   text_auto=True)
    chart5.update_layout(barmode='stack', xaxis={'categoryorder': 'total descending'})
    graphJSON5 = json.dumps(chart5, cls=plotly.utils.PlotlyJSONEncoder)
    sauske['GC5'] = graphJSON5
    
    # grouping
    df6 = ds.groupby(['Shift'])['DurInMin'].sum().reset_index(name ='DurInMin')
    # plot the dataframe
    #df5.plot(x='Shift', y=['DurInMin'], kind="bar", figsize=(9, 8))
    #sauske['Shift'] = '/static/figure5.png'
    #mp.savefig(os.path.join('static', "figure5.png"))  # mp.show()
    chart6 = px.bar(df6,
                   x='Shift',
                   y='DurInMin',
                   color_discrete_sequence = ['#DEE71F']*len(df6),
                   template= 'plotly_white',
                   text_auto=True)
    chart6.update_layout(barmode='stack', xaxis={'categoryorder': 'total descending'})
    graphJSON6 = json.dumps(chart6, cls=plotly.utils.PlotlyJSONEncoder)
    sauske['GC6'] = graphJSON6

    
    

    df7 = ds.groupby(['Group code','Cause code']).size().nlargest(10).reset_index(name='FreqCount')

    chart7 = px.scatter(df7,
                        x='Group code',
                        y='Cause code',
                        color_discrete_sequence=['purple'] * len(df7),
                        template='plotly_white')
    chart7.update_layout(xaxis={'categoryorder': 'total descending'})
    chart7.update_layout(showlegend=True)  
    #chart7.update_layout(visibility='hidden')  # Hide the chart initially
    graphJSON7 = json.dumps(chart7, cls=plotly.utils.PlotlyJSONEncoder)
    sauske['GC7'] = graphJSON7
    
    return sauske



    
    
    
    
    

