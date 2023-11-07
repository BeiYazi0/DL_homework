from cnn.models import load_model
from cnn.utils.visualization import confusion_show
import numpy as np
from scipy.io import loadmat

data = loadmat('data/MNISTData.mat')

X_train = data['X_Train']
X_test = data['X_Test']
y_train = data['D_Train'].astype(np.int32)
y_test = data['D_Test'].astype(np.int32)

X_train = np.expand_dims(X_train.transpose(2,0,1), axis=1)
X_test = np.expand_dims(X_test.transpose(2,0,1), axis=1)
y_train = y_train.T
y_test = y_test.T

model = load_model("model\\standard_MINST_cnn.h5")

print(f"模型在训练集上的表现\n{model.evaluate(X_train, y_train)}")
print(f"模型在测试集上的表现\n{model.evaluate(X_test, y_test)}")
labels = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
confusion_show(labels, model.predict_classes(X_test), y_test.argmax(axis = 1))
