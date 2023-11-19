from cnn.models import load_model
from scipy.io import loadmat

data = loadmat('data/POI.mat')
data = data['SquPOI'].T
data_mean = data.mean(axis = 0)
data -= data_mean

X_train = data[:600]
y_train = data[1:601]
X_test = data[600:-1]
y_test = data[601:]

model = load_model("model\\standard_POI_rnn.h5")


print(f"模型在测试集上的表现\n{model.evaluate(X_test, y_test)}")
