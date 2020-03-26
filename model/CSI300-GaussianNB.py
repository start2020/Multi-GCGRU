```
from keras import Input, layers, models, callbacks, utils
import numpy as np
from keras import backend as K
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

# import data
path = "./data/model input/CSI300-features.xlsx"
df = pd.read_excel(path)

len_dates = len(list(df['trade_date'].unique()))
codes = list(df['ts_code'].unique())
len_code = len(codes)
print(len_dates)
print(len_code)
print(df.shape)

n = len_code # 股票支数
time_steps = 3 # 时间序列
features_num = 6 # 特征个数
positive = 0 # 趋势阈值
choose_num = 10 # 所选择的股票数

# labels
# 制造全部标签
labels = []
for c in codes[:choose_num]:
    df_code = df[df['ts_code']==c] # 某支股票
    label = df_code['labels-close'].values[time_steps:] # 第time_steps个时刻开始
    labels.extend(label) # 共315700
labels = np.array(labels)
print(labels.shape)

# samples
import numpy as np
df = df.sort_values(by=['trade_date'])
df2 = df[['open','high','close','low','vol','amount']] # 被纳入考虑的标签
samples = df2.apply(lambda x:(x-np.min(x)) / (np.max(x)-np.min(x)))
# 检验是否将每个都标准化
print(samples['open'].min(),samples['high'].min())

samples = []
for i in range(time_steps, len_dates):
    times = []
    for j in range(i-time_steps,i):
        x = df2.iloc[j*len_code:(j+1)*len_code].values #时刻i的第j个训练样本，每个样本维度是(287,6)
        times.append(x)
    times = np.array(times)
    samples.append(times)
samples = np.array(samples)
print(samples.shape)
samples = np.tile(samples,(choose_num,1,1,1)) # 多个股票对应的数量
print(samples.shape)

samples_num = samples.shape[0]
samples = np.reshape(samples, (samples_num,3*287*6))
print(samples.shape)

# training set and test set
train_num = 9000
x_train = samples[:train_num]
y_train = labels[:train_num]
x_test = samples[train_num:]
y_test = labels[train_num:]
print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)

############################ model
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(x_train,y_train)

############################ predict
y2 = model.predict(x_test)
y1 = y_test

############################ Cacualtion cost
from sklearn.metrics import roc_auc_score, accuracy_score,precision_score,confusion_matrix, recall_score, f1_score,matthews_corrcoef
res = confusion_matrix(y1,y2)
acc = accuracy_score(y1, y2)
precision = precision_score(y1,y2)
recall = recall_score(y1,y2)
f1score = f1_score(y1,y2)
mcc = matthews_corrcoef(y1,y2)
print(res)
print('accuracy_score is :', acc)
print('precision_score is  : ',precision)
print('recall_score is  : ' ,recall)
print('f1_score is : ',f1score)
print('matthews_corrcoef is : ',mcc)

############################### Confusion Matrix
y_true = y1
y_pred = y2
labels = ['A', 'B']
 
tick_marks = np.array(range(len(labels))) + 0.5
cm = confusion_matrix(y_true, y_pred)
def plot_confusion_matrix(cm, title='Confusion Matrix', cmap=plt.cm.binary):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(labels)))
    plt.xticks(xlocations, labels, rotation=90)
    plt.yticks(xlocations, labels)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    #cm = confusion_matrix(y_true, y_pred)
    np.set_printoptions(precision=2)

cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print(cm_normalized)
plt.figure(figsize=(12, 8), dpi=120)
 
ind_array = np.arange(len(labels))
x, y = np.meshgrid(ind_array, ind_array)
 
for x_val, y_val in zip(x.flatten(), y.flatten()):
    c = cm_normalized[y_val][x_val]
    if c > 0.01:
        plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=7, va='center', ha='center')
# offset the tick
plt.gca().set_xticks(tick_marks, minor=True)
plt.gca().set_yticks(tick_marks, minor=True)
plt.gca().xaxis.set_ticks_position('none')
plt.gca().yaxis.set_ticks_position('none')
plt.grid(True, which='minor', linestyle='-')
plt.gcf().subplots_adjust(bottom=0.15)
 
plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')
# show confusion matrix
#plt.savefig('../Data/confusion_matrix.png', format='png')
plt.show()
```

