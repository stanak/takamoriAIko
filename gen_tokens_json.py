from janome.tokenizer import Tokenizer
import glob
import json

csvs = glob.glob('data/*.csv')
aiko_quotes = []
for csv in csvs:
    with open(csv, 'r', encoding='shift_jis') as f:
        quotes = [line.split(',')[1] for line in f.read().splitlines()]
        while '' in quotes:
            quotes.remove('')
        aiko_quotes.extend(quotes)

tokenizer = Tokenizer("userdic.csv", udic_enc="shift_jis")
tokens = []
for q in aiko_quotes:
    tokens.append([str(token) for token in tokenizer.tokenize(q)])
with open('tokens.json', 'w', encoding='shift_jis') as f:
    json.dump(tokens, f, ensure_ascii=False, indent=2)
