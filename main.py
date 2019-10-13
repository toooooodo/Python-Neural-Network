from dnn import DNN

input_n = 3
hidden_n = 3
output_n = 3
learning_rate = 0.5


def main():
    model = DNN(input_n, hidden_n, output_n, learning_rate)
    # print(model.train([1, 2, 3]))


if __name__ == '__main__':
    main()
