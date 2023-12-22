from nltk import wordpunct_tokenize


class Vocab:
    def __init__(self, max_length, get_year, vocab = None, data = None):
        self.max_length = max_length
        self.data = data
        self.vocab = {}
        self.get_year = get_year
        if self.data is not None:
            self.build_vocab()
        else:
            self.vocab = vocab

    def build_vocab(self):
        
        set_key = [v for title in self.data for v in wordpunct_tokenize(title.lower())]
        set_key = set(set_key)
        self.vocab["sos"] = 0
        self.vocab["pad"] = 1
        self.vocab["eos"] = 2
        self.vocab["unk"] = 3
        for i, v in enumerate(set_key):
            self.vocab[v.lower()] = 4 + i

    def word_encode(self, text):
        ws = wordpunct_tokenize(text.lower())
        s = [0]
        for w in ws:
            if not self.get_year and w.is_digit():
                continue

            if w != "(" and w != ")":
                if w.lower in self.vocab.keys():
                    s.append(self.vocab[w.lower()])
                else:
                    s.append(self.vocab["unk"])

        s.append(self.vocab["eos"])

        while len(s) < self.max_length:
            s.append(self.vocab["pad"])

        return s  
    def size(self):
        return len(self.vocab)