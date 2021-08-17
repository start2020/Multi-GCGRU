```
from keras import Input, layers, models, callbacks, utils
import numpy as np
from keras import backend as K
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

########################################### Datasets
path = ".../data/model input/CSI300-features.xlsx"
df = pd.read_excel(path,index_col=0)
# 标准化样本
df1 = df[['open','high','close','low','vol','amount']]
df1 = df1.apply(lambda x:(x-np.min(x)) / (np.max(x)-np.min(x)))
df[['open','high','close','low','vol','amount']] = df1
len_dates = len(list(df['trade_date'].unique())) # 总天数
codes = list(df['ts_code'].unique()) # 股票代码
len_code = len(codes) #股票数
pre = 3 # 取前3个时刻
choose_num = 24000
# 制造全部标签
labels = []
for c in codes:
    df_code = df[df['ts_code']==c] # 某支股票
    label = df_code['labels-close'].values[pre:] # 第pre个时刻开始
    labels.extend(label) # 共315700
labels = np.array(labels)
print(labels.shape)
labels = labels[:choose_num]
print(labels.shape)

# 制造全部样本
samples = []
for c in codes[0:]:
    df_code = df[df['ts_code']==c][['open','high','close','low','vol','amount']] # 某支股票
    num = df_code.shape[0]
    for i in range(num-pre):
        x = df_code.iloc[i:i+pre].values
        samples.append(x)
samples = np.array(samples)
print(samples.shape)
feature_num = 6
samples = samples.reshape(samples.shape[0], pre*feature_num)
print(samples.shape)
# 缩小一半
samples = samples[:choose_num]
print(samples.shape)

# 随机分割训练集和测试集
from sklearn.model_selection import train_test_split  # 更新
x_train, x_test, y_train, y_test = train_test_split(samples, labels, test_size=0.2)
print(x_train.shape, y_train.shape)

####################################### Train Model
from sklearn import svm  #导入sklearn模块中svm
model = svm.SVC(gamma=0.1,C=100)    #定义一个svm分类器
model.fit(x_train, y_train)   #用训练集构建模型

###################################### Accuracy Caculation
y2 = model.predict(x_test)
y1 = y_test
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

###################################### Confusion Matrix 
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
plt.show()
```

