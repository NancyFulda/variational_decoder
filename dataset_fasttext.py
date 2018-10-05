from collections import Counter
import numpy as np
import codecs
import torch
from torch.autograd import Variable


class Dataset:


    def __init__(self, filename="data/ijcnlp_dailydialog/dialogues_text.txt"):
        self.generate_vocab(filename)
        self.generate_conversations(filename)
        self.create_batches()
        self.reset_batch_pointer()


    def generate_vocab(self, filename):
        self.vocab_size = 50002 #number of fastText vectors + SOS and EOS
        self.max_sentence_length = 30

        #with codecs.open(filename, "r", encoding='utf-8') as f:
        #    data = f.read()
        #counter = Counter(data)
        #count_pairs = sorted(counter.items(), key=lambda x: -x[1])
        #self.chars, _ = zip(*count_pairs)
        #self.chars = list(self.chars)

        #self.chars.append("SOS ")
        #self.chars.append(" EOS")

        #self.vocab_size = len(self.chars)
        #self.vocab = dict(zip(self.chars, range(len(self.chars))))
        #self.EOS_index = self.vocab[" EOS"]
        #self.SOS_index = self.vocab["SOS "]

    
    def generate_conversations(self, filename):
        with open(filename, "r") as file:
            text_content = file.readlines()
            
        self.conversations = []
        for line_no in range(len(text_content)):
            text = text_content[line_no].split("__eou__")
            text = [t.strip() for t in text if t.strip() != ""]

            for turn in range(len(text) - 1):
                self.conversations.append((text[turn], text[turn+1]))
    
                #truncate sentences that are too long?
                #if len(text[turn].split(' ')) < self.max_sentence_length:
                #    if len(text[turn+1].split(' ')) < self.max_sentence_length:
                #        self.conversations.append((text[turn], text[turn+1]))


    def create_batches(self):
        self.encoder_turns = []
        self.decoder_turns = []
        self.encoded_conversations = []
        longest_possible_sequence = 200
        for convo in self.conversations:
            if len(convo[0]) > longest_possible_sequence or \
                    len(convo[1]) > longest_possible_sequence:
                continue

            #encoder_turn = list(map(self.vocab.get, convo[0]))
            encoder_turn = convo[0].lower().split(' ')
            #encoder_turn.insert(0, self.SOS_index)
            #encoder_turn.append(self.EOS_index)
            encoder_turn.insert(0, 'SOS')
            encoder_turn.append('EOS')
            encoder_turn = np.array(encoder_turn)

            #decoder_turn = list(map(self.vocab.get, convo[1]))
            decoder_turn = convo[1].lower().split(' ')
            #decoder_turn.insert(0, self.SOS_index)
            #decoder_turn.append(self.EOS_index)
            decoder_turn.insert(0, 'SOS')
            decoder_turn.append('EOS')
            decoder_turn = np.array(decoder_turn)

            self.encoded_conversations.append((encoder_turn, decoder_turn))
            self.encoder_turns.append(encoder_turn)
            self.decoder_turns.append(decoder_turn)


    def next_batch(self):
        encoder, decoder = self.encoder_turns[self.pointer], self.decoder_turns[self.pointer]
        self.pointer += 1
        if self.pointer >= len(self.encoder_turns):
            self.reset_batch_pointer()
        return encoder, decoder


    def reset_batch_pointer(self):
        self.pointer = 0


    def to_onehot(self, x, long_type=False):
        onehot_stack = torch.zeros((len(x), self.vocab_size))
        onehot_stack[np.array(range(len(x))), x] = 1
        if long_type:
            onehot_stack = onehot_stack.type(torch.LongTensor)
        return Variable(onehot_stack)


    def to_phrase(self, x):
        return "".join([self.chars[x[i]] for i in range(len(x))])
    
    def to_array(self, x):
        #return np.array([self.chars.index(char) for char in x])
        return np.array(x.lower().split(' '))

    def size(self):
        return len(self.encoder_turns)



if __name__ == "__main__":
    dataset = Dataset()
    x, y = dataset.next_batch()

    # print the indicies
    print(x)
    print(y)

    # print the chars
    print("".join([dataset.chars[x[i]] for i in range(len(x))]))
    print("".join([dataset.chars[y[i]] for i in range(len(y))]))
