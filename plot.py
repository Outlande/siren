import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data = np.genfromtxt('/home/yan/Work/Threads/siren/results/res_accuracy.txt')
data2 = np.genfromtxt('/home/yan/Work/Threads/siren/results/fin_accuracy.txt')
#
#plt.rc('font', family='cmss17')
#print(np.append(data, data2))
df = pd.DataFrame({
    "Method": ["Ours"]*len(data)+["Finetune"]*len(data),
    "Updated Model" : np.tile([10, 60, 110, 160, 210, 260, 310], 2),
    "Accuracy": np.append(data, data2)
})
#print(df)
sns.pointplot(x='Updated Model', y='Accuracy', hue='Method', markers=["o", "x"], linestyles=["-", "--"], err_style="band", ci=95, data=df)
#sns.scatterplot(x='Frame', y='Accuracy', hue='Method', ci=None, data=df)
sns.set(font_scale=5)
sns.set()
plt.show()
# data.columns=['x']
# data['y']=None
# for i in range(len(data)):
#   coordinate=data['x'][i].split()
#   data['x'][i]=coordinate[0]
#   data['y'][i]=coordinate[1]
# print(data)
# sns.lineplot(data=data, x='x', y='y')