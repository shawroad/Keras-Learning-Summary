"""

@file   : 2-新样本的预测.py

@author : xiaolu

@time1  : 2019-06-10

"""
import cv2
from keras.models import load_model
import numpy as np


# 读取测试图片 并显示图片
img = cv2.imread('./test.png')
# cv2.imshow("test", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

print(img.shape)    # 输出(100, 100, 3) 很明显是三通道的图片

# 接下来转灰度图 并缩放到28*28
# 1. 转灰度
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(gray.shape)   # 输出(100, 100)

# 2. 缩放到28*28
image = cv2.resize(gray, (28, 28))
print(image.shape)   # 输出 (28, 28)

# 此时，我们将图片整理好了
image = image.astype('float32').reshape((-1, 28, 28, 1))   # 这里这一步 是为了迎合模型的输入

image = image / 255.

# 加载模型
model = load_model('mnist_LeNet5.h5')
y_pred = model.predict(image)
y_pred = np.argmax(y_pred)
print("最终的预测结果为:", y_pred)






