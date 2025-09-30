import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

benchmarks = pd.read_csv('adbench/result/benchmarks.csv')
msde = pd.read_csv('adbench/result/MSML2.csv')
msde['model'] = 'MSDE'

def extract_aucroc(x):
    try:
        return eval(x)['aucroc']
    except NameError:
        return np.nan

def extract_aucpr(x):
    try:
        return eval(x)['aucpr']
    except NameError:
        return np.nan

benchmarks['index'] = benchmarks.apply(lambda x: eval(x['setting']) + (x['model'],), axis=1)
benchmarks = benchmarks.set_index('index').drop(columns=['setting', 'model'])
benchmarks.index = pd.MultiIndex.from_tuples(benchmarks.index)

msde['index'] = msde.apply(lambda x: eval(x['setting']) + (x['model'],), axis=1)
msde = msde.set_index('index').drop(columns=['setting', 'model'])
msde.index = pd.MultiIndex.from_tuples(msde.index)
benchmarks = pd.concat([benchmarks, msde]).sort_index().dropna()
# print(benchmarks.columns)

benchmarks['aucroc'] = benchmarks['results'].apply(extract_aucroc)
benchmarks['aucpr'] = benchmarks['results'].apply(extract_aucpr)
benchmarks.drop(columns=['results'], inplace=True)

benchmarks = benchmarks.groupby(level=(0, 3)).mean().swaplevel(0, 1).sort_index()
models = benchmarks.index.get_level_values(0).unique()

aucrocArrays = []
aucprArrays = []
timeArrays = []

for model in models:
    aucrocArrays.append(benchmarks.loc[model, 'aucroc'].values)
    aucprArrays.append(benchmarks.loc[model, 'aucpr'].values)
    timeArrays.append(benchmarks.loc[model, 'time_fit'].values + benchmarks.loc[model, 'time_inference'].values)

plt.figure(figsize=(15, 10))
plt.boxplot(aucrocArrays, tick_labels=models)
plt.title('Area under the ROC curve')
plt.savefig('aucroc.svg')
plt.clf()
plt.boxplot(aucprArrays, tick_labels=models)
plt.title('Area under the precision-recall curve')
plt.savefig('aucpr.svg')
plt.clf()
plt.boxplot(timeArrays, tick_labels=models)
plt.title('Time')
plt.yscale('log')
plt.savefig('time.svg')
plt.clf()
