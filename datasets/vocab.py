import nltk

nltk.download('punkt')

class Vocab:
    def __init__(self, max_length, data, get_year):
        self.max_length = max_length
        self.data = data
        self.vocab = {}
        self.get_year = get_year
        self.build_vocab()

    def build_vocab(self):
        
        set_key = [v for title in self.data for v in nltk.word_tokenize(title)]
        set_key = set(set_key)
        self.vocab["sos"] = 0
        self.vocab["pad"] = 1
        self.vocab["eos"] = 2
        for i, v in enumerate(set_key):
            self.vocab[v.lower()] = 3 + i

    def word_encode(self, text):
        ws = nltk.word_tokenize(text)
        s = [0]
        for w in ws:
            if not self.get_year and w.is_digit():
                continue

            if w != "(" and w != ")":
                s.append(self.vocab[w.lower()])

        s.append(self.vocab["eos"])

        while len(s) < self.max_length:
            s.append(self.vocab["pad"])

        return s  