"""

@file   : 08-简单多层感知机预测疾病-画损失 准确率.py

@author : xiaolu

@time   : 2019-06-14

"""
from sklearn.datasets import load_breast_cancer
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

breast = load_breast_cancer()
X = breast.data
Y = breast.target
print(X.shape)   # (569, 30)
print(Y.shape)   # (569,)


# 我们用keras实现几层

model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(30, )))
model.add(Dropout(0.2))
model.add(Dense(32, activation='tanh'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

epochs = 100
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

file_path = '乳腺癌预测.h5'
checkpoint = ModelCheckpoint(filepath=file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')


history = model.fit(X, Y, batch_size=5, epochs=epochs, validation_split=0.2, callbacks=[checkpoint])


plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title("model acc")
plt.xlabel("epoch")
plt.ylabel('acc')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("model loss")
plt.xlabel("epoch")
plt.ylabel('loss')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()


