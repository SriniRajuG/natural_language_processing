import os
import sys
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import utils
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model


def get_word_embeddings_map(glove_dir):
    # Build a dictionary mapping words to their embedding vectors
    # Word vectors from Glove are being used.
    word_embeddings_map = {}
    fo = open(os.path.join(glove_dir, 'glove.6B.50d.txt'))
    for line in fo:
        values = line.split()
        word = values[0]
        embedding_vec = np.asarray(values[1:], dtype='float32')
        word_embeddings_map[word] = embedding_vec
    fo.close()
    print('Found %s word vectors.' % len(word_embeddings_map))
    return word_embeddings_map


def read_text_data(text_data_dir):
    texts = []  # list of strings. Each element of this list is a document.
    labels_int_map = {}
    # A dictionary mapping label names (class names) to integer ids. Starting with 0
    labels = []  # list of integers
    # *texts* and *labels* will be of the same size.
    for dir_name in sorted(os.listdir(text_data_dir)):
        # loop over each directory (type of news, 0 to 19)
        # *dir_name* is label name / class name
        dir_path = os.path.join(text_data_dir, dir_name)
        if os.path.isdir(dir_path):
            label_id = len(labels_int_map)  # label_id is the integer corresponding to a class
            labels_int_map[dir_name] = label_id
            for file_name in sorted(os.listdir(dir_path)):
                if file_name.isdigit():
                    file_path = os.path.join(dir_path, file_name)
                    if sys.version_info < (3,):
                        fo = open(file_path)
                    else:
                        fo = open(file_path, encoding='latin-1')
                    txt = fo.read()
                    i = txt.find('\n\n')  # skip header
                    if 0 < i:
                        txt = txt[i:]
                    texts.append(txt)
                    fo.close()
                    labels.append(label_id)
    print('no. of documents', len(texts))
    print('no. of labels', len(labels_int_map))
    return texts, labels, labels_int_map


def texts_to_seqs(texts, max_num_words, max_seq_len):
    # vectorize the text samples into a 2D integer tensor. All sequences are padded to be of the
    #   same length
    tokenizer = Tokenizer(num_words=max_num_words)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded_seqs = pad_sequences(sequences, maxlen=max_seq_len)
    print('Shape of text-data tensor:', padded_seqs.shape)
    word_index_map = tokenizer.word_index
    return padded_seqs, word_index_map


def split_train_test(padded_seqs, labels, validation_perct):
    # split the data into a training set and a validation set
    n_texts = padded_seqs.shape[0]
    indices = np.arange(n_texts)
    # shuffling texts and labels
    np.random.shuffle(indices)
    padded_seqs = padded_seqs[indices]
    labels = labels[indices]
    num_validation_samples = int(validation_perct * n_texts)
    x_train = padded_seqs[: -num_validation_samples]
    y_train = labels[: -num_validation_samples]
    x_val = padded_seqs[-num_validation_samples:]
    y_val = labels[-num_validation_samples:]
    return x_train, y_train, x_val, y_val


def create_embedding_matrix(word_index_map, word_embeddings_map, max_num_words, embedding_dim):
    print('Preparing embedding matrix.')
    n_uniq_tokens = len(word_index_map)
    num_words = min(max_num_words, n_uniq_tokens)
    embeddings_matrix = np.zeros((num_words, embedding_dim))
    for word, i in word_index_map.items():
        if i >= max_num_words:
            continue
        embedding_vector = word_embeddings_map.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embeddings_matrix[i] = embedding_vector
    return embeddings_matrix


def cnn_model(n_words, embedding_dim, max_seq_len, embedding_matrix, n_labels):
    embedding_layer = Embedding(input_dim=n_words, output_dim=embedding_dim,
                                weights=[embedding_matrix], input_length=max_seq_len,
                                trainable=False)
    # embedding_matrix.shape[0] and the argument for input_dim must be the same.
    # Load pre-trained word embeddings into an Embedding layer
    # We set trainable = False so as to keep the embeddings fixed
    # Using functional API
    # Train a 1D convnet with global maxpooling
    sequence_input = Input(shape=(max_seq_len,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    x = Conv1D(filters=128, kernel_size=5, activation='relu')(embedded_sequences)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = GlobalMaxPooling1D()(x)
    x = Dense(128, activation='relu')(x)
    preds = Dense(n_labels, activation='softmax')(x)
    model = Model(inputs=sequence_input, outputs=preds)
    return model


def main():

    GLOVE_DIR = '/home/osboxes/PycharmProjects/learn_neural_net/NLP/glove_data'
    TEXT_DATA_DIR = '/home/osboxes/PycharmProjects/learn_neural_net/NLP/text_data/20_newsgroup'
    MAX_SEQUENCE_LENGTH = 1000
    MAX_NUM_WORDS = 20000
    EMBEDDING_DIM = 50
    VALIDATION_SPLIT = 0.2

    word_embeddings_map = get_word_embeddings_map(GLOVE_DIR)
    texts, labels, labels_int_map = read_text_data(TEXT_DATA_DIR)
    n_labels = len(labels_int_map)
    padded_seqs, word_index_map = texts_to_seqs(texts, MAX_NUM_WORDS, MAX_SEQUENCE_LENGTH)
    labels = utils.to_categorical(np.asarray(labels))  # labels is not a list of integers anymore.
    print('Shape of label tensor:', labels.shape)
    x_train, y_train, x_val, y_val = split_train_test(padded_seqs, labels, VALIDATION_SPLIT)
    n_uniq_tokens = len(word_index_map)
    n_words = min(MAX_NUM_WORDS, n_uniq_tokens)
    embeddings_matrix = create_embedding_matrix(word_index_map, word_embeddings_map, MAX_NUM_WORDS,
                                                EMBEDDING_DIM)
    model = cnn_model(n_words, EMBEDDING_DIM, MAX_SEQUENCE_LENGTH, embeddings_matrix, n_labels)
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])
    model.fit(x_train, y_train, batch_size=128, epochs=10, validation_data=(x_val, y_val))


if __name__ == '__main__':
    main()