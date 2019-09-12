"""

@file  : 攻击模型InceptionV3.py

@author: xiaolu

@time  : 2019-09-11

"""
import numpy as np
from keras.preprocessing import image
from keras.applications import inception_v3
from keras import backend as K
from PIL import Image


def demo1():
    # 演示梯度下降求解的过程  损失函数为 y=x2+2
    import random
    a = 0.1
    x = random.randint(1, 10)
    y = x * x + 2
    index = 1
    while index < 100 and abs(y-2) > 0.01 :
        y = x*x+2
        print("batch={} x={} y={}".format(index,x,y))
        x = x-2*x*a
        index += 1


def demo2():
    # 演示使用现成的模型进行判断
    from keras.applications.resnet50 import preprocess_input, decode_predictions
    model = inception_v3.InceptionV3()
    # img = image.load_img("./data/pig.jpg", target_size=(299, 299))  # 加载图片
    img = image.load_img("./hacked-pig-image.png", target_size=(299, 299))  # 加载伪造的图片
    original_image = image.img_to_array(img)  # 将图片变为矩阵

    # 对图片进行预处理
    original_image /= 255.
    original_image -= 0.5
    original_image *= 2.

    original_image = np.expand_dims(original_image, axis=0)
    # print(original_image.shape)   # (1, 299, 299, 3)

    preds = model.predict(original_image)
    print('Predicted:', decode_predictions(preds, top=3)[0])   # 输出前三个概率最大的类别


def demo3():
    # 演示使用现成的模型进行判断
    from keras.applications.resnet50 import preprocess_input, decode_predictions
    from keras.applications.resnet50 import ResNet50

    model = ResNet50(weights='imagenet')

    img_path = 'hacked-pig_image.png'
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    # decode the results into a list of tuples (class, description, probability)
    # (one such list for each sample in the batch)
    print('Predicted:', decode_predictions(preds, top=3)[0])


def demo4():
    # 将猪变为面包机
    model = inception_v3.InceptionV3()
    model_input_layer = model.layers[0].input
    model_output_layer = model.layers[-1].output

    object_type_to_fake = 859  # 烤面包机对应的标签
    # 加载图片　并进行预处理
    img = image.load_img("./data/pig.jpg", target_size=(299, 299))
    original_image = image.img_to_array(img)
    original_image /= 255.
    original_image -= 0.5
    original_image *= 2.
    original_image = np.expand_dims(original_image, axis=0)

    # 为了避免肉眼超过肉眼可接受的成都,我们需要设定阈值
    max_change_above = original_image + 0.01
    max_change_below = original_image - 0.01

    hacked_image = np.copy(original_image)  # numpy处于性能的考虑 赋值只是相当于引用 所以我们这里必须手动拷贝

    learning_rate = 0.1

    cost_function = model_output_layer[0, object_type_to_fake]  # 将猪预测成面包机?

    gradient_function = K.gradients(cost_function, model_input_layer)[0]  # 传进损失函数　获取输入部分的梯度

    # K.learning_phase() == 0: 代表训练模式,  == 1: 代表测试模式
    grab_cost_and_gradients_from_model = K.function([model_input_layer, K.learning_phase()], [cost_function, gradient_function])
    cost = 0.0
    index = 1
    # 我们认为烤面包机的概率超过60%即可, 所以我们可以定义损失函数的值超过0.6即可以完成训练
    while cost < 0.60:
        cost, gradients = grab_cost_and_gradients_from_model([hacked_image, 0])
        hacked_image += gradients * learning_rate
        hacked_image = np.clip(hacked_image, max_change_below, max_change_above)  # 改变的时候　不能让幅度太大　不然人眼就认不出是猪了
        hacked_image = np.clip(hacked_image, -1.0, 1.0)
        print("batch:{} Cost: {:.8}%".format(index, cost * 100))
        index += 1

    img = hacked_image[0]
    img /= 2.
    img += 0.5
    img *= 255.

    im = Image.fromarray(img.astype(np.uint8))
    im.save("hacked-pig-image.png")


def demo5():

    model = inception_v3.InceptionV3()
    model_input_layer = model.layers[0].input
    model_output_layer = model.layers[-1].output
    object_type_to_fake = 859
    img = image.load_img("./data/pig.jpg", target_size=(299, 299))
    original_image = image.img_to_array(img)

    original_image /= 255.
    original_image -= 0.5
    original_image *= 2.

    original_image = np.expand_dims(original_image, axis=0)

    max_change_above = original_image + 0.01
    max_change_below = original_image - 0.01

    hacked_image = np.copy(original_image)

    cost_function = model_output_layer[0, object_type_to_fake]
    gradient_function = K.gradients(cost_function, model_input_layer)[0]

    grab_cost_and_gradients_from_model = K.function([model_input_layer, K.learning_phase()], [cost_function, gradient_function])

    cost = 0.0
    index = 1
    learning_rate = 0.3
    e = 0.007

    while cost < 0.99:
        cost, gradients = grab_cost_and_gradients_from_model([hacked_image, 0])
        # fast gradient sign method
        # EXPLAINING AND HARNESSING ADVERSARIAL EXAMPLES
        # hacked_image += gradients * learning_rate
        n = np.sign(gradients)   # 将梯度进行了一个符号函数  大于0 设置为1 小于0 设置为-1 等于0 设置为0
        hacked_image += n * e
        # print gradients

        hacked_image = np.clip(hacked_image, max_change_below, max_change_above)
        hacked_image = np.clip(hacked_image, -1.0, 1.0)

        print("batch:{} Cost: {:.8}%".format(index, cost * 100))
        index += 1

    img = hacked_image[0]
    img /= 2.
    img += 0.5
    img *= 255.

    im = Image.fromarray(img.astype(np.uint8))
    im.save("hacked-pig-image.png")


if __name__ == '__main__':
    # 1. 简单了解梯度下降
    # demo1()

    # 2. 用inceptionV3进行图像预测
    demo2()

    # 3. 用ResNet50进行图像预测
    # demo3()

    # 4.
    # demo4()

    # 5.
    # demo5()




