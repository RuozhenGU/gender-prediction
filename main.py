import pandas as pd
import numpy as np
from model import RNN
import random

def read_process():
    """
    Read tsv dataset and process as json output

    Return: two json key value pairs for train and test data 
    """

    def process_data(data):
        j = {}
        for index, row in data.iterrows():
            j[row["Person Name"]] = row["Gender"]
        return j

    # read in dataset
    data = pd.read_csv(r'allnames.tsv', sep='\t')

    # process gender data
    data["Gender"] = data["Gender"].replace(({"Male": 0, "Female": 1}))

    # split train test
    train_data = data[data["Train/Test"] == "Train"]
    test_data = data[data["Train/Test"] == "Test"]

    # format as json
    train_data = process_data(train_data)
    test_data = process_data(test_data)

    return train_data, test_data

def vocal_mapping(train_data):
    """
    Get vocal size insights

    Args:
        train_data (df): json object with name as key, 0/1 as value
    """

    vocab = list(set([w for text in train_data.keys() for w in text.split(' ')]))
    vocab_size = len(vocab)

    print('%d unique words found' % vocab_size)

    # Assign indices to each word.
    word_to_idx = { w: i for i, w in enumerate(vocab) }
    idx_to_word = { i: w for i, w in enumerate(vocab) }

    return word_to_idx, idx_to_word, vocab_size

def one_hot_encode(text, vocab_size, word_to_idx):
    inputs = []
    for w in text.split(' '):
        v = np.zeros((vocab_size, 1))
        v[word_to_idx[w]] = 1
        inputs.append(v)
    return inputs

def softmax(xs):
  # Applies the Softmax Function to the input array.
  return np.exp(xs) / sum(np.exp(xs))

def train(rnn, data, vocal_size, word_to_idx, backprop=True):
    items = list(data.items())
    random.shuffle(items)

    loss = 0
    num_correct = 0

    for x, y in items:
        inputs = one_hot_encode(x, vocal_size, word_to_idx)
        target = int(y)

        # Forward
        out, _ = rnn.forward(inputs)
        probs = softmax(out)

        # loss / accuracy
        loss -= np.log(probs[target])
        num_correct += int(np.argmax(probs) == target)

        if backprop:
            d_L_d_y = probs
            d_L_d_y[target] -= 1
            rnn.backprop(d_L_d_y)

    return loss / len(data), num_correct / len(data)

def main(epoch=500):

    # get dataset
    train_data, test_data = read_process()

    # get mapping
    word_to_idx, idx_to_word, vocal_size = vocal_mapping(train_data)

    # init rnn model class
    rnn = RNN(vocal_size, 2)

    # train
    for epoch in range(epoch):

        print("Training start ...")

        train_loss, train_acc = train(rnn, train_data, vocal_size, word_to_idx)
        
        print('--- Epoch %d' % (epoch + 1))

        print('Train:\tLoss %.3f | Accuracy: %.3f' % (train_loss, train_acc))

        test_loss, test_acc = train(rnn, test_data, vocal_size, word_to_idx, backprop=False)
        
        print('Test:\tLoss %.3f | Accuracy: %.3f' % (test_loss, test_acc))
