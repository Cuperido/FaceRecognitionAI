# This is a script to call UWImg lab5 code.
# It doesn't include any UWImg codes for security reasons.

from uwimg import *

def softmax_model(inputs, outputs):
    l = [make_layer(inputs, outputs, SOFTMAX)]
    return make_model(l)

def neural_net(inputs, outputs):
    l = [make_layer(inputs, 64, RELU),
         make_layer(64, 32, RELU),
         make_layer(32, outputs, SOFTMAX)]
    return make_model(l)

print("loading data...")
train = load_classification_data(b"v1s/v1s.train", b"v1s/labels.txt", 1)
print("loading data...")
test = load_classification_data(b"v1s/v1s.test", b"v1s/labels.txt", 1)
print("done")
print

print("training model...")
batch = 128
iters = 1000
rate = 0.01
momentum = .9
decay = 0.001

m = neural_net(train.X.cols, train.y.cols)
train_model(m, train, batch, iters, rate, momentum, decay)
print("done")
print

print("evaluating model...")
print("training accuracy: %f", accuracy_model(m, train))
print("test accuracy:     %f", accuracy_model(m, test))
