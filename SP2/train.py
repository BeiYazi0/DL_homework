from cnn.layers import *
from cnn.models import Model
from cnn.utils.visualization import history_show
from scipy.io import loadmat
import time

data = loadmat('data/POI.mat')
data = data['SquPOI'].T
data_mean = data.mean(axis = 0)
data -= data_mean

X_train = data[:600]
y_train = data[1:601]
X_test = data[600:-1]
y_test = data[601:]


input_layer = Input((3, ))
model = Model(input_layer, "POI_rnn")

# 添加网络层
model.add_layer(Recurrent(10, 3, activate_fcn = "tanh"))
model.add_layer(Output(3, 10, activate_fcn = "Linear"))

model.compile(0.015, "MSE", "MAE")

#model.summary()


T1 = time.time()
history = model.fit(X_train, y_train, batch_size = 50, epochs = 18, verbose = 2, shuffle = False)
T2 = time.time()
print('训练用时:%s秒' % ((T2 - T1)))

model.save("model\\POI_rnn.h5")

print(f"模型在测试集上的表现\n{model.evaluate(X_test, y_test)}")
history_show(history)

