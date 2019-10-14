from dnn import DNN
import numpy as np

layers = [28 * 28, 200, 10]
learning_rate = 0.1
epochs = 5
train_data_set = 'mnist_train.csv'
test_data_set = 'mnist_test.csv'


def main():
    model = DNN(layers, learning_rate)

    # train
    with open(train_data_set, 'r', encoding='utf-8') as f:
        train_data_list = f.readlines()
        print(f"train data: {len(train_data_list)}")
    for epoch in range(epochs):
        for record in train_data_list:
            all_values = record.split(',')
            inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
            targets = np.zeros(layers[-1]) + 0.01
            targets[int(all_values[0])] = 0.99
            model.train(inputs, targets)
        print(f"epoch {epoch}")

    # test
    with open(test_data_set, 'r', encoding='utf-8') as f:
        test_data_list = f.readlines()
        print(f"test data: {len(test_data_list)}")
    score_card = list()
    for record in test_data_list:
        all_values = record.split(',')
        inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        outputs = model.forward(inputs)
        predict = np.argmax(outputs)
        label = int(all_values[0])
        if predict == label:
            score_card.append(1)
        else:
            score_card.append(0)
    print(f"test accuracy = {sum(score_card) / len(test_data_list)}")


if __name__ == '__main__':
    main()
