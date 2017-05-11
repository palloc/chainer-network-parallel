#!/usr/bin/env python
import numpy as np
import six
import chainer
import chainer.functions as F
import chainer.links as L
from chainer.datasets import cifar
from chainer import cuda
from chainer import optimizers
from chainer import function


BATCH_SIZE = 64
N_EPOCH = 30
GPU1 = 0
GPU2 = 1
CLASS_LABELS = 10


class NetworkParallel:
    def __init__(self):
        pass

    # Prepare dataset(MNIST)
    def read_dataset(self):
        print('load cifar100 dataset')
        cifar100_train, cifar100_test = cifar.get_cifar10()

        # Create train data
        self.x_train = []
        self.y_train = []
        for i in cifar100_train:
            self.x_train.append(i[0])
            self.y_train.append(i[1])
        self.x_train = np.array(self.x_train)
        self.y_train = np.array(self.y_train)

        # Create test data
        self.x_test = []
        self.y_test = []
        for i in cifar100_test:
            self.x_test.append(i[0])
            self.y_test.append(i[1])
        self.x_test = np.array(self.x_test)
        self.y_test = np.array(self.y_test)

        # Size of datasets
        self.N = self.y_train.size
        self.N_test = self.y_test.size
        print (self.x_train.shape)


    def define_model(self):
        # Model1
        self.model = chainer.FunctionSet(
            conv1_1=L.Convolution2D(3, 64, 3, stride=1, pad=1),
            conv1_2=L.Convolution2D(None, 64, 3, stride=1, pad=1),
            bn1=L.BatchNormalization(64),

            conv2_1=L.Convolution2D(None, 128, 3, stride=1, pad=1),
            conv2_2=L.Convolution2D(None, 128, 3, stride=1, pad=1),
            bn2=L.BatchNormalization(128),

            conv3_1=L.Convolution2D(None, 256, 3, stride=1, pad=1),
            conv3_2=L.Convolution2D(None, 256, 3, stride=1, pad=1),
            conv3_3=L.Convolution2D(None, 256, 3, stride=1, pad=1),
            conv3_4=L.Convolution2D(None, 256, 3, stride=1, pad=1),
            bn3=L.BatchNormalization(256),
            
            fc6=L.Linear(None, 1024),
            fc7=L.Linear(None, 1024),
            fc8=L.Linear(None, CLASS_LABELS))


        # Model2
        self.model2 = chainer.FunctionSet(
            conv1_1=L.Convolution2D(3, 64, 3, stride=1, pad=1),
            conv1_2=L.Convolution2D(None, 64, 3, stride=1, pad=1),
            bn1=L.BatchNormalization(64),

            conv2_1=L.Convolution2D(None, 128, 3, stride=1, pad=1),
            conv2_2=L.Convolution2D(None, 128, 3, stride=1, pad=1),
            bn2=L.BatchNormalization(128),

            conv3_1=L.Convolution2D(None, 256, 3, stride=1, pad=1),
            conv3_2=L.Convolution2D(None, 256, 3, stride=1, pad=1),
            conv3_3=L.Convolution2D(None, 256, 3, stride=1, pad=1),
            conv3_4=L.Convolution2D(None, 256, 3, stride=1, pad=1),
            bn3=L.BatchNormalization(256),

            fc6=L.Linear(None, 1024),
            fc7=L.Linear(None, 1024),
            fc8=L.Linear(None, CLASS_LABELS))

        # Assign models to GPU
        self.model.to_gpu(GPU1)
        self.model2.to_gpu(GPU2)

    # Neural net architecture
    def forward(self, x_data, y_data, train=True):
        x, t = chainer.Variable(x_data), chainer.Variable(y_data)

        h = F.relu(self.model.bn1(self.model.conv1_1(x)))
        h = F.relu(self.model.bn1(self.model.conv1_2(h)))
        h = F.dropout(h, ratio=0.25, train=train)
        h = F.max_pooling_2d(h, 2)

        h = F.relu(self.model.bn2(self.model.conv2_1(h)))
        h = F.relu(self.model.bn2(self.model.conv2_2(h)))
        h = F.dropout(h, ratio=0.25, train=train)
        h = F.max_pooling_2d(h, ksize=2, stride=2)

        h = F.relu(self.model.bn3(self.model.conv3_1(h)))
        h = F.relu(self.model.bn3(self.model.conv3_2(h)))
        h = F.relu(self.model.bn3(self.model.conv3_3(h)))
        h = F.relu(self.model.bn3(self.model.conv3_4(h)))
        h = F.dropout(h, ratio=0.25, train=train)
        h = F.max_pooling_2d(h, 2, stride=2)

        # Change GPU1 --> GPU2
        h.data = cuda.to_gpu(h.data, device=GPU2)

        h = F.dropout(F.relu(self.model2.fc6(h)), train=train, ratio=0.25)
        h = F.dropout(F.relu(self.model2.fc7(h)), train=train, ratio=0.25)
        y = self.model2.fc8(h)

        self.loss, self.acc = F.softmax_cross_entropy(y, t), F.accuracy(y, t)

    # Setting learning options and start learning
    def learning(self):
        # Setup optimizer
        self.optimizer = optimizers.Adam()
        self.optimizer.setup(self.model)

        # Learning loop
        for epoch in six.moves.range(1, N_EPOCH+1):
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
                  .format(sum_loss/self.N, sum_accuracy/self.N))

            # Evaluation
            self.perm = np.random.permutation(self.N_test)
            sum_accuracy = 0
            sum_loss = 0
            for i in six.moves.range(0, self.N_test, BATCH_SIZE):
                self.x_batch = self.x_test[self.perm[i:i + BATCH_SIZE]]
                self.y_batch = self.y_test[self.perm[i:i + BATCH_SIZE]]
                # Reshape data for GPU
                self.x_batch = cuda.to_gpu(self.x_batch, device=GPU1)
                self.y_batch = cuda.to_gpu(self.y_batch, device=GPU2)

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
