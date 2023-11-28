from cnn.models import load_model
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

data = loadmat('data/POI.mat')
data = data['SquPOI'].T
data_mean = data.mean(axis = 0)
data -= data_mean

X_train = data[:600]
y_train = data[1:601]
X_test = data[600:-1]
y_test = data[601:]

model = load_model("model\\special_POI_rnn.h5")

y_single = model.predict(X_test)

fig = plt.figure()
fig.suptitle('single predict')

t = np.linspace(1, 100, 100)                # 创建t的取值范围

ax1 = fig.add_subplot(3, 1, 1) 
ax1.plot(t, y_single[:100, 0] + data_mean[0], label='predict_x')
ax1.plot(t, y_test[:100, 0] + data_mean[0], label='true_x')
ax1.set_xlabel('time')
ax1.set_ylabel('value')
ax1.legend()

ax2 = fig.add_subplot(3, 1, 2)  
ax2.plot(t, y_single[:100, 1] + data_mean[1], label='predict_y')
ax2.plot(t, y_test[:100, 1] + data_mean[1], label='true_y')
ax2.set_xlabel('time')
ax2.set_ylabel('value')
ax2.legend()

ax3 = fig.add_subplot(3, 1, 3)    
ax3.plot(t, y_single[:100, 2] + data_mean[2], label='predict_z')
ax3.plot(t, y_test[:100, 2] + data_mean[2], label='true_z')
ax3.set_xlabel('time')
ax3.set_ylabel('value')
ax3.legend()

plt.show()
