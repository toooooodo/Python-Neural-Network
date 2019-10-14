from dnn import DNN
import numpy as np
import random

layers = [28 * 28, 256, 10]
learning_rate = 0.1
epochs = 5
train_data_set = 'mnist_train.csv'
test_data_set = 'mnist_test.csv'

model = DNN(layers, learning_rate)
np.random.seed(12345)
random.seed(12345)


def main():
    with open(train_data_set, 'r', encoding='utf-8') as f:
        train_data_list = f.readlines()
        print(f"train data: {len(train_data_list)}")
        train_data = []
        for record in train_data_list:
            all_values = record.split(',')
            inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
            targets = np.zeros(layers[-1]) + 0.01
            targets[int(all_values[0])] = 0.99
            train_data.append((inputs, targets))
    with open(test_data_set, 'r', encoding='utf-8') as f:
        test_data_list = f.readlines()
        print(f"test data: {len(test_data_list)}")
        test_data = []
        for record in test_data_list:
            all_values = record.split(',')
            inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
            label = int(all_values[0])
            test_data.append((inputs, label))
    # train
    for epoch in range(epochs):
        random.shuffle(train_data)
        loss = sum(model.train(inputs, targets) for inputs, targets in train_data)
        print(f"epoch {epoch}, loss = {loss / len(train_data_list)}")
        # test
        correct_num = 0
        for inputs, label in test_data:
            outputs = model.forward(inputs)
            predict = np.argmax(outputs)
            correct_num += int(label == predict)
        print(f"epoch {epoch}, test accuracy = {correct_num / len(test_data_list)}")


if __name__ == '__main__':
    main()
