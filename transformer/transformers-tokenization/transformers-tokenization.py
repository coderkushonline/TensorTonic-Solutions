class SimpleTokenizer:
    def __init__(self):
        self.word_to_id = {}
        self.id_to_word = {}
        self.vocab_size = 0
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"

    def build_vocab(self, texts: list[str]) -> None:
        """
        Builds the vocabulary in place.
        """
        self.word_to_id = {self.pad_token : 0, self.unk_token : 1, self.bos_token : 2, self.eos_token : 3}

        i = 4
        nested = [text.split(' ') for text in texts]
        flat = sorted([item.lower() for sublist in nested for item in sublist])
        for word in flat:
            if word not in self.word_to_id:
                self.word_to_id[word] = i
                i += 1
        

        self.id_to_word = {value:key for key,value in self.word_to_id.items()}
        self.vocab_size = i 

    def encode(self, text: str) -> list[int]:
        """
        Returns token IDs for the input text.
        """
        encoded = []
        text_list = text.lower().split(' ')
        for word in text_list:
            if word in self.word_to_id:
                encoded.append(self.word_to_id[word])
            elif len(word) != 0:
                encoded.append(self.word_to_id[self.unk_token])
        return encoded
        
    def decode(self, ids: list[int]) -> str:
        """
        Returns the decoded, space-separated text.
        """
        decoded = []
        for id in ids:
            if id in self.id_to_word:
                decoded.append(self.id_to_word[id])
            else:
                decoded.append(self.unk_token)

        return " ".join(decoded)