"""
Identify the character-vocabulary (set of unique characters) from the documents. Lets say that the
  character-vocab-size = 60 i,e 60 unique characters in the data-set
Create one-hot-vectors (OHV) for each character in the document.
Because we are using CNNs all the docs should be of the same length. Pad all the documents at the
  end with zeros until their length is equal to the longest doc in the data-set. Or, pad all the
  documents to a pre-determined value. Say, *max_length* = 300.
If n_docs = 1000, n_chars_in_each_doc = 300, character_vocab_size = 60; The shape of input tensor is
  (n_docs, n_characters_in_each_doc, character_vocab_size) i,e. (1000, 300, 60)


Convolutional NN are being used. Functional API of Keras.
"""


import pandas as pd
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers import Conv1D, MaxPooling1D


def docs_to_charOHVs(docs, char_idx_map, maxlen):
    # OHV is one-hot-vector
    vocab_size = len(char_idx_map)
    X = []
    for doc in docs:
        char_indices = [char_idx_map[char] for char in doc]
        OHVs = np.eye(vocab_size)[char_indices]
        X.append(OHVs)
    # Only of *maxlen* is larger than the largest document, use it to pad sequences. Otherwise
    #   pad all the sequences to the length of largest sequence.
    return pad_sequences(X, maxlen=maxlen)


def preproc_data(reviews_df):
    reviews = reviews_df.loc[:, 'review']
    sentiments = reviews_df.loc[:, 'label']
    docs = list()
    labels = list()
    for document, sentiment in zip(reviews, sentiments):
        doc_lower_case = [char.lower() for char in document]
        docs.append(doc_lower_case)
        labels.append(sentiment)
    # Identify unique characters
    txt = ''
    for doc in docs:
        for char in doc:
            txt += char
    uniq_chars = set(txt)
    char_idx_map = dict((char, idx_char) for idx_char, char in enumerate(uniq_chars))
    # idx_char_map = dict((idx_char, char) for idx_char, char in enumerate(uniq_chars))
    return docs, labels, char_idx_map


def conv1d_model(input_shape):
    n_filters = 256
    dense_outputs = 1024
    filter_size = [7, 7, 3, 3, 3, 3]
    n_out = 2
    inputs = Input(shape=input_shape, name='input', dtype='float32')
    conv0 = Conv1D(filters=n_filters, kernel_size=filter_size[0], padding='valid',
                   activation='relu', input_shape=input_shape)(inputs)
    conv0 = MaxPooling1D(pool_size=3)(conv0)
    conv1 = Conv1D(filters=n_filters, kernel_size=filter_size[1], padding='valid',
                   activation='relu')(conv0)
    conv1 = MaxPooling1D(pool_size=3)(conv1)
    conv2 = Conv1D(filters=n_filters, kernel_size=filter_size[2], padding='valid',
                   activation='relu')(conv1)
    conv3 = Conv1D(filters=n_filters, kernel_size=filter_size[3], padding='valid',
                   activation='relu')(conv2)
    conv4 = Conv1D(filters=n_filters, kernel_size=filter_size[4], padding='valid',
                   activation='relu')(conv3)
    conv5 = Conv1D(filters=n_filters, kernel_size=filter_size[5], padding='valid',
                   activation='relu')(conv4)
    conv5 = MaxPooling1D(pool_size=3)(conv5)
    conv5 = Flatten()(conv5)
    z = Dropout(0.5)(Dense(units=dense_outputs, activation='relu')(conv5))
    z = Dropout(0.5)(Dense(units=dense_outputs, activation='relu')(z))
    pred = Dense(units=n_out, activation='softmax', name='output')(z)
    model = Model(input=inputs, output=pred)
    return model


def main():
    reviews_df = pd.read_csv("/home/osboxes/PycharmProjects/learn_neural_net/NLP/text_data/"
                       "sentiment_labelled_sentences/yelp_labelled.txt", delimiter='\t',
                       header=None, names=['review', 'label'], encoding='utf-8')
    docs, labels, char_idx_map = preproc_data(reviews_df)
    print(len(docs))
    vocab_size = len(char_idx_map)
    print('total chars:', vocab_size)

    maxlen = 1024
    train_data = docs_to_charOHVs(docs, char_idx_map, maxlen)
    # OHV is one-hot-vector
    y_train = to_categorical(labels)

    print(train_data.shape)
    # input to CNN should be of the shape (num_docs, max_length_of_docs, vocab_size)

    input_shape = (maxlen, vocab_size)
    model = conv1d_model(input_shape)
    batch_size = 80
    nb_epoch = 10
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop',  metrics=['accuracy'])
    model.summary()

    # model.fit(train_data, y_train, batch_size=32, nb_epoch=120, validation_split=0.2,
    # verbose=False)


if __name__ == '__main__':
    main()
