```
from keras import Input, layers, models, callbacks, utils
import numpy as np
from keras import backend as K
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

##########################################  Datasets
path = ".../data/model input/CSI500-features.xlsx"
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
choose_num = 15 # 所选择的股票数
# 制造全部标签
labels = []
for c in codes[:choose_num]:
    df_code = df[df['ts_code']==c] # 某支股票
    label = df_code['labels-close'].values[time_steps:] # 第time_steps个时刻开始
    labels.extend(label) # 共315700
labels = np.array(labels)
print(labels.shape)

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
# RNN
samples_num = samples.shape[0]
samples = np.reshape(samples, (samples_num,3,489*6))
print(samples.shape)
# 随机分割训练集和测试集
from sklearn.model_selection import train_test_split  # 更新
x_train, x_test, y_train, y_test = train_test_split(samples, labels, test_size=0.2)
print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)
print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)

############################################### Model Train
# 输入
input0 = Input(shape=(samples.shape[1],samples.shape[2]))
input1 = layers.Flatten()(input0)
input2 = layers.Dense(100)(input1)
output = layers.Dense(1, activation='sigmoid')(input2)
model = models.Model(inputs=[input0],outputs=[output])
model.summary()
model.compile(loss='binary_crossentropy',optimizer='sgd',metrics=['accuracy'])
callbacks_list = [
    callbacks.ModelCheckpoint(filepath='ann-csi500.h5',
                                    monitor='val_loss',
                                    save_best_only=True,)]
model.summary()
# 训练
epochs = 10
interval_batch = 500
H = model.fit(x_train, y_train,callbacks=callbacks_list,batch_size=interval_batch,epochs=epochs,validation_data=(x_test,y_test))

#############################################  Accuracy Caculation
path = "./ann-csi500.h5"
model_best = models.load_model(path)
predictions = model_best.predict(x_test)
y2 = []
for i in range(len(predictions)):
    if predictions[i][0] >= 0.5:
        y2.append(1)
    else:
        y2.append(0)
y1 = y_test
y2 = np.array(y2)
print(y1,y2)
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

#############################################  Confusion Matrix
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

