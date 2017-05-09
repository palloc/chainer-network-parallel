#!/usr/bin/env python
import numpy as np
import six
import data
import chainer
import chainer.functions as F
from chainer import cuda
from chainer import optimizers
from chainer import function


BATCH_SIZE = 100
N_EPOCH = 20
GPU1 = 0
GPU2 = 1


class NetworkParallel:
    def __init__(self):
        pass

    # Prepare dataset(MNIST)
    def read_dataset(self):
        print('load MNIST dataset')
        mnist = data.load_mnist_data()
        mnist['data'] = mnist['data'].astype(np.float32)
        mnist['data'] /= 255
        mnist['target'] = mnist['target'].astype(np.int32)

        # Number of data
        self.N = 60000
        self.x_train, self.x_test = np.split(mnist['data'],   [self.N])
        self.y_train, self.y_test = np.split(mnist['target'], [self.N])
        # Number of test data
        self.N_test = self.y_test.size

    def define_model(self):
        # Model1
        self.model = chainer.FunctionSet(conv1=F.Convolution2D(1, 20, 5),
                                         conv2=F.Convolution2D(20, 50, 5),
                                         l1=F.Linear(200, 1000),
                                         l2=F.Linear(1000, 1000))

        # Model2
        self.model2 = chainer.FunctionSet(l1=F.Linear(1000, 10000),
                                          l2=F.Linear(10000, 10))

        # Assign models to GPU
        self.model.to_gpu(GPU1)
        self.model2.to_gpu(GPU2)

    # Neural net architecture
    def forward(self, x_data, y_data, train=True):
        x_data = x_data.reshape((len(x_data), 1, 28, 28))
        x, t = chainer.Variable(x_data), chainer.Variable(y_data)

        h = F.max_pooling_2d(F.relu(self.model.conv1(x)),
                             ksize=6,
                             stride=3,
                             pad=1)
        h = F.max_pooling_2d(F.relu(self.model.conv2(h)), 2)
        h = F.dropout(F.relu(self.model.l1(h)),  train=train)
        h = F.relu(self.model.l2(h))
        # Change GPU1 --> GPU2
        h.data = cuda.to_gpu(h.data, device=GPU2)
        h = F.dropout(F.relu(self.model2.l1(h)), train=train)
        y = self.model2.l2(h)

        self.loss, self.acc = F.softmax_cross_entropy(y, t), F.accuracy(y, t)

    # Setting learning options and start learning
    def learning(self):
        # Setup optimizer
        self.optimizer = optimizers.Adam()
        self.optimizer.setup(self.model.collect_parameters())

        # Learning loop
        for epoch in six.moves.range(1, N_EPOCH + 1):
            print('epoch', epoch)

            # Training
            self.perm = np.random.permutation(self.N)
            sum_accuracy = 0
            sum_loss = 0

            for i in six.moves.range(0, self.N, BATCH_SIZE):
                self.x_batch = self.x_train[self.perm[i:i + BATCH_SIZE]]
                self.y_batch = self.y_train[self.perm[i:i + BATCH_SIZE]]
                # Reshape data for GPU
                self.x_batch = cuda.to_gpu(self.x_batch, device=GPU1)
                self.y_batch = cuda.to_gpu(self.y_batch, device=GPU2)
                # Forward
                self.optimizer.zero_grads()
                self.forward(self.x_batch, self.y_batch)
                # Backward
                self.loss.backward()
                self.optimizer.update()
                # Calc loss and accuracy
                sum_loss += float(cuda.to_cpu(self.loss.data))*len(self.y_batch)
                sum_accuracy += float(cuda.to_cpu(self.acc.data))*len(self.y_batch)

            print('train mean loss={}, accuracy={}'
                  .format(sum_loss / self.N, sum_accuracy / self.N))

            # Evaluation
            sum_accuracy = 0
            sum_loss = 0
            for i in six.moves.range(0, self.N_test, BATCH_SIZE):
                # Forward
                self.forward(self.x_batch, self.y_batch, train=False)
                # Calc loss and accuracy
                sum_loss += float(cuda.to_cpu(self.loss.data))*len(self.y_batch)
                sum_accuracy += float(cuda.to_cpu(self.acc.data))*len(self.y_batch)

            print('test  mean loss={}, accuracy={}'
                  .format(sum_loss/self.N_test, sum_accuracy/self.N_test))


def main():
    test = NetworkParallel()
    test.read_dataset()
    test.define_model()
    test.learning()


if __name__ == '__main__':
    main()
