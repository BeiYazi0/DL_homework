from cnn.layers import *
from cnn.models import Model, load_model
from cnn.utils.visualization import history_show, confusion_show
import numpy as np
from scipy.io import loadmat
import time

data = loadmat('data/MNISTData.mat')

X_train = data['X_Train']
X_test = data['X_Test']
y_train = data['D_Train'].astype(np.int32)
y_test = data['D_Test'].astype(np.int32)

X_train = np.expand_dims(X_train.transpose(2,0,1), axis=1)
X_test = np.expand_dims(X_test.transpose(2,0,1), axis=1)
y_train = y_train.T
y_test = y_test.T

input_layer = Input((1, 28, 28))
model = Model(input_layer, "MNIST_cnn")

# 添加网络层
model.add_layer(Conv2D(20, 9, input_shape = (1, 28, 28), activate_fcn = "ReLU"))
model.add_layer(AveragePooling2D(2, input_shape = (20, 20, 20)))
model.add_layer(Flatten((20, 10, 10)))
model.add_layer(Dense(100, 2000, activate_fcn = "ReLU"))
model.add_layer(Output(10, 100))

model.compile(0.01, "cross_tropy")

T1 = time.time()
history = model.fit(X_train, y_train, batch_size = 32, epochs = 3, verbose = 1, shuffle = True)
T2 = time.time()
print('训练用时:%s分' % ((T2 - T1) / 60))

print(f"模型在测试集上的表现\n{model.evaluate(X_test, y_test)}")
history_show(history)
labels = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
confusion_show(labels, model.predict_classes(X_test), y_test.argmax(axis = 1))

model.save("model\\MNIST_cnn.h5")
