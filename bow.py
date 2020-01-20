#!/usr/bin/python3

from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import RMSprop
import numpy as np
import random
import json
import sys


class Token:
    """
    surface (表層形)
    part_of_speech (品詞,品詞細分類1,品詞細分類2,品詞細分類3のリスト)
    base_form (基本形)
    infl_form (活用形)
    infl_type (活用型)
    reading (読み)
    phonetic (発音)
    """
    def __init__(self, record: str) -> None:
        self.record = record
        self.surface, data = record.split('\t')
        self.part_of_speech = data.split(',')[:4]
        (
            self.base_form,
            self.infl_form,
            self.infl_type,
            self.reading,
            self.phonetic
        ) = data.split(',')[4:]

    def __eq__(self, other):
        if not isinstance(other, Token):
            return NotImplemented
        return self.record == other.record

    def __str__(self):
        return self.record

    def __repr__(self):
        return self.record

    def __hash__(self):
        return hash(self.record)


class Tokens:
    def __init__(self) -> None:
        self.freq = {}
        self.list = []
        self.indices = {}
        self.indices_token = {}

    @classmethod
    def from_json(cls, path: str, encoding='shift_jis') -> 'Tokens':
        index_count = 0
        with open(path, 'r', encoding=encoding) as f:
            all_tokens = json.load(f)
        res = Tokens()
        for tokens in all_tokens:
            for record in tokens:
                token = Token(record)
                res.list.append(token)
                if token in res.freq:
                    res.freq[token] += 1
                else:
                    res.freq[token] = 1
                    res.indices[token] = index_count
                    index_count += 1
        res.indices_token = dict([(value, key)
                                  for (key, value) in res.indices.items()])
        return res


tokens = Tokens.from_json('tokens.json')

# cut the text in semi-redundant sequences of maxlen characters
maxlen = 5
step = 1
sentences = []
next_words = []
for i in range(0, len(tokens.list) - maxlen, step):
    sentences.append(tokens.list[i: i + maxlen])
    next_words.append(tokens.list[i + maxlen])
print('nb sequences:', len(sentences))

print('Vectorization...')
x = np.zeros((len(sentences), maxlen, len(tokens.list)), dtype=np.bool)
y = np.zeros((len(sentences), len(tokens.list)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, token in enumerate(sentence):
        x[i, t, tokens.indices[token]] = 1
    y[i, tokens.indices[next_words[i]]] = 1


# build the model: a single LSTM
print('Build model...')
model = Sequential()
model.add(LSTM(128, input_shape=(maxlen, len(tokens.list))))
model.add(Dense(len(tokens.list), activation='softmax'))

optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def on_epoch_end(epoch, _):
    # Function invoked at end of each epoch. Prints generated text.
    print()
    print('----- Generating text after Epoch: %d' % epoch)

    start_index = random.randint(0, len(tokens.list) - maxlen - 1)
    start_index = 0  # テキストの最初からスタート
    for diversity in [0.2]:  # diversity は 0.2のみ使用
        print('----- diversity:', diversity)

        generated = ''
        sentence = tokens.list[start_index: start_index + maxlen]
        # sentence はリストなので文字列へ変換して使用
        generated += ''.join(list(map(lambda x: x.surface, sentence)))
        print('----- Generating with seed: "' +
              ''.join(list(map(lambda x: x.surface, sentence))) + '"')
        sys.stdout.write(generated)

        for i in range(400):
            x_pred = np.zeros((1, maxlen, len(tokens.list)))
            for j, token in enumerate(sentence):
                x_pred[0, j, tokens.indices[token]] = 1.

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_token = tokens.indices_token[next_index]

            generated += next_token.surface
            sentence = sentence[1:]
            # sentence はリストなので append で結合する
            sentence.append(next_token)

            sys.stdout.write(next_token.surface)
            sys.stdout.flush()
        print()


print_callback = LambdaCallback(on_epoch_end=on_epoch_end)

model.fit(x, y,
          batch_size=128,
          epochs=60,
          callbacks=[print_callback])
