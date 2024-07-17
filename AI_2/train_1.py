import pickle
from sklearn.metrics import  precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.ensemble as ske
from keras.callbacks import History
from keras.callbacks import LearningRateScheduler
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.src.layers import BatchNormalization
from keras.src.optimizers import Adam
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

data = pd.read_csv('train_result1.csv', sep=';')
X1 = data.drop(['type'], axis=1).values
y = data['type'].values

print('Researching important feature based on %i total features\n' % X1.shape[1])
# Feature selection using Trees Classifier
fsel = ske.ExtraTreesClassifier().fit(X1, y)

model1 = SelectFromModel(fsel, prefit=True)
X_new = model1.transform(X1)
nb_features = X_new.shape[1]

features = []

print('%i features identified as important:' % nb_features)

xnew = [0] * nb_features
indices = np.argsort(fsel.feature_importances_)[::-1][:nb_features]
for f in range(nb_features):
    xnew[f] = data.columns[2 + indices[f]]
    print(" %d. feature %s (%f)" % (f + 1, data.columns[2 + indices[f]], fsel.feature_importances_[indices[f]]))

res = {}
for f in range(X1.shape[1]):
    res[data.columns[f]] = 0
for f in range(nb_features):
    res[xnew[f]] = 1

i = 0
xd = [0] * (X1.shape[1] - nb_features + 1)
for f in range(X1.shape[1]):
    if res[data.columns[f]] == 0:
        xd[i] = data.columns[f]
        i = i + 1
xd[i] = 'type'
xfinal = data.drop(xd, axis=1).values
X_train, X_test, y_train, y_test = train_test_split(xfinal, y, test_size=0.2, random_state=0)
# 假设你有一个 History 对象，记录了训练过程中的损失和指标
history = History()

# 定义学习率调度函数
def lr_scheduler(epoch, lr):
    if epoch > 0 and epoch % 100 == 0:
        return lr * 0.5
    else:
        return lr

# 创建 Adam 优化器，设置初始学习率
optimizer = Adam(learning_rate=0.001)

model = Sequential()
# 输入层和第一个密集层（将S维度转换为512维度）
model.add(Dense(units=512, input_dim=nb_features, activation='relu'))

# 批归一化和Dropout进行正则化
model.add(BatchNormalization())
model.add(Dropout(rate=0.2))

# 重复批归一化、Dropout和密集层的序列
for _ in range(2):  # 示例中重复3次，根据需要调整
    model.add(Dense(units=512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(rate=0.2))

model.add(Dense(units=1, activation='sigmoid'))
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# 定义学习率调度器
#lr_scheduler_callback = LearningRateScheduler(lr_scheduler)
# 训练模型，传入学习率调度器作为回调函数
model.fit(X_train, y_train, batch_size=18, epochs=500, callbacks=[history])
model.save('train.keras')
model.save_weights("train_weighs.keras")
open('features.pkl', 'wb').write(pickle.dumps(xnew))
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.6)

# 计算mt
mt = confusion_matrix(y_test, y_pred)

# 绘制热力图
plt.figure(figsize=(8, 6))
sns.set(font_scale=1.2)  # 设置字体大小

mt=confusion_matrix(y_test,y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
# 绘制热力图
sns.heatmap(mt, annot=True, fmt='d', cmap='Blues', cbar=False,
            annot_kws={"size": 14}, linewidths=0.5, linecolor='grey')

# 添加标题和标签
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')

# 显示图表
plt.show()

# 绘制训练集和验证集的损失曲线
plt.plot(history.history['loss'], label='Training Loss')
plt.ylabel('Loss')

# 显示图表
plt.legend()
plt.show()

# 示例数据：精确度、召回率和 F1 分数
metrics = ['Precision', 'Recall', 'F1 Score']
scores = [precision, recall, f1]

# 创建条形图
plt.figure(figsize=(8, 6))
plt.bar(metrics, scores, color=['blue', 'green', 'orange'])
plt.xlabel('Metrics')
plt.ylabel('Score')
plt.title('Model Performance Metrics')
plt.ylim(0.0, 1.0)  # 确保y轴范围适合精确度、召回率和F1分数的范围
plt.grid(True)

# 在条形图上显示数值
for i, score in enumerate(scores):
    plt.text(i, score + 0.02, f'{score:.2f}', ha='center', va='bottom', fontsize=12)

plt.show()