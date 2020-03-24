```
from keras import Input, layers, models, callbacks, utils
import numpy as np
from keras import backend as K
import tensorflow as tf

########################################  Datasets
import pandas as pd
path = ".../data/model input/CSI300-features.xlsx"
df = pd.read_excel(path)

len_dates = len(list(df['trade_date'].unique()))
codes = list(df['ts_code'].unique())
len_code = len(codes)
print(len_dates)
print(len_code)
print(df.shape)

# labels
c = codes[2]
print(c)
df1 = df[df['ts_code']==c]
num = df1.shape[0]
train_num = 900 # 训练集
pre = 3 # 取前3个时刻
labels = utils.to_categorical(df1['multi-labels'].values) 
labels_train = labels[pre:train_num+pre]
labels_test = labels[train_num+pre:]
print(labels_train.shape)
print(labels_test.shape)

# samples
import numpy as np
df = df.sort_values(by=['trade_date'])
df2 = df[['open','high','close','low','vol','amount']] # 被纳入考虑的标签
samples = df2.apply(lambda x:(x-np.min(x)) / (np.max(x)-np.min(x)))
# 检验是否将每个都标准化
print(samples['open'].min(),samples['high'].min())

samples_train = []
samples_test = []
for i in range(len_code):
    stock_train = []
    stock_test = []
    for j in range(1103*i,1103*(i+1)-pre):
        x = samples.iloc[j:j+pre].values
        if j-1103*i<train_num:
            stock_train.append(x)
        else:
            stock_test.append(x)
    stock_train = np.array(stock_train)
    samples_train.append(stock_train)
    stock_test = np.array(stock_test)
    samples_test.append(stock_test)

####################################### Model
n = len_code # 股票数
time_steps = pre # 时间序列
features_num = 6 # 特征个数
def fun(x):
    from keras import Input, layers, models, callbacks, utils
    import numpy as np
    from keras import backend as K
    import tensorflow as tf
    import pandas as pd
    path = ".../data/model input/CSI300-shareholder-matrix.xlsx"
    # 287 is the number of stocks
    adjacency_concept = pd.read_excel(path,index_col=0).values.reshape((1,287,287)) 
    adjacency_concepts  = tf.convert_to_tensor(adjacency_concept,dtype=np.float32)
    y = K.batch_dot(x,adjacency_concepts,axes=(1,2))
    y = K.permute_dimensions(y, (0,2,1))
    return y
 
def gcn_layer(units, inputs):
    gcn1 = layers.Lambda(fun)(inputs)
    gcn2 = layers.Dense(units,activation='tanh')(gcn1)
    return gcn2
    
end_lstm = 1
rnn1 = layers.SimpleRNN(300,return_sequences=True)
rnn2 = layers.SimpleRNN(end_lstm)

inputs = []
outputs = []
for i in range(n):
    x1 = Input(shape=(time_steps,features_num))
    inputs.append(x1)
    x2 = rnn1(x1)
    x3 = rnn2(x2)
    out = layers.Reshape((1,end_lstm))(x3)
    outputs.append(out)
merge =  layers.concatenate(outputs,axis=1)

gcn1 = gcn_layer(30,merge)
gcn2 = gcn_layer(1,gcn1)
gcn3 = layers.Flatten()(gcn2)
output = layers.Dense(1, activation='sigmoid')(out)

model = models.Model(inputs,output)
#model.summary()
model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])
callbacks_list = [
    callbacks.ModelCheckpoint(filepath='rnn-gcns.h5',
                                    monitor='val_loss',
                                    save_best_only=True,)]
                                    
############################################ Train
epochs = 100
interval_batch = 10
H = model.fit(samples_train, labels_train,callbacks=callbacks_list,batch_size=interval_batch,epochs=epochs,validation_data=(samples_test, labels_test))


########################################### Accuracy caculation
path = "./rnn-gcns.h5"
model_best = models.load_model(path)
predictions = model_best.predict(labels_test)
y2 = []
for i in range(len(predictions)):
    if predictions[i][0] >= 0.5:
        y2.append(1)
    else:
        y2.append(0)
y1 = labels_test
y2 = np.array(y2)
print(y1,y2)

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

