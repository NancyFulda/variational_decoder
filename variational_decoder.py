import numpy as np
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import pyro
import pyro.optim as optim
import pyro.distributions as dist
from pyro.infer import SVI
from pyro.util import ng_zeros, ng_ones

import fasttext
from dataset import Dataset
import sys

import os
os.environ["TFHUB_CACHE_DIR"] = "/mnt/pccfs/not_backed_up/hub/"
import tensorflow as tf
import tensorflow_hub as hub

#initialize sentence embeddings
#g1 = tf.Graph()
#with g1.as_default():
#    #embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/2")
#    embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder-lite/2")

#    session = tf.Session()
#    session.run([tf.global_variables_initializer(), tf.tables_initializer()])

#
# ===============================================
# ===============================================
# ===============================================
#
# Globals


# init dataset
dataset = Dataset()

# experimental variables
LOSS_METHOD = 'ELBO'
INPUT_EMBEDDING = 'bag_of_words' #bag of words embedding using fastText
WORD_EMBEDDING = 'FastText'
USE_NEGATIONS = False

# Sizes
if WORD_EMBEDDING == 'FastText':
    VOCAB_SIZE = 50002

if INPUT_EMBEDDING == 'bag_of_words':
    SENTENCE_EMBEDDING_SIZE = 300
    OUTER_LSTM_HIDDEN_SIZE = 300
    DECODER_HIDDEN_SIZE = 600
    Z_DIMENSION = 300 
elif INPUT_EMBEDDING == 'use_lite':
    SENTENCE_EMBEDDING_SIZE = 512
    OUTER_LSTM_HIDDEN_SIZE = 512
    DECODER_HIDDEN_SIZE = 1024
    Z_DIMENSION = 300 
elif INPUT_EMBEDDING == 'use_large':
    SENTENCE_EMBEDDING_SIZE = 512
    OUTER_LSTM_HIDDEN_SIZE = 512
    DECODER_HIDDEN_SIZE = 1024
    Z_DIMENSION = 300 

LEARNING_RATE = .0001
MAX_LENGTH = 300
NUM_LAYERS_FOR_RNNS = 1
#CONTEXT_LENGTH = 5

USE_CUDA = False
TEACHER_FORCING = True


#
# ===============================================
# ===============================================
# ===============================================
#
# RNNs

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, ftext):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.fc = nn.Linear(input_size, num_layers*hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size, num_layers=num_layers)
        
        self.fc.weight.data.copy_(torch.FloatTensor(ftext.vectors).view_as(self.fc.weight.data))
        #self.fc.weight.requires_grad = False


    def forward(self, input_var, hidden):
        if USE_CUDA:
            input_var = input_var.cuda()

        with g1.as_default():
            embedded = self.fc(input_var).view(self.num_layers, 1, -1)
        output, hidden = self.rnn(embedded, hidden)

        if type(self.fc.weight.grad) == type(None):
            print("EncoderRNN fc gradiants are none")

        if type(self.rnn.weight_ih_l0.grad) == type(None):
            print("EncoderRNN IH gradiants are none")
        if type(self.rnn.weight_hh_l0.grad) == type(None):
            print("EncoderRNN HH gradiants are none")

        return output, hidden

    def init_hidden_lstm(self):
        result = (Variable(torch.zeros(self.num_layers, 1, self.hidden_size)),
                  Variable(torch.zeros(self.num_layers, 1, self.hidden_size)))
        if USE_CUDA:
            return result.cuda()
        else:
            return result

    def init_hidden_gru(self):
        result = Variable(torch.zeros(self.num_layers, 1, self.hidden_size))
        if USE_CUDA:
            return result.cuda()
        else:
            return result


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Linear(output_size, num_layers*hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size, num_layers=num_layers)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_var, hidden):
        if USE_CUDA:
            input_var = input_var.cuda()

        hidden = hidden.view(self.num_layers, 1, self.hidden_size)

        output = self.embedding(input_var).view(self.num_layers, 1, -1)
        output = F.relu(output)
        output, hidden = self.rnn(output, hidden)
        output = self.softmax(self.out(output[0]))

        if type(self.embedding.weight.grad) == type(None):
            print("DecoderRNN embedding weights are none")
        if type(self.rnn.weight_ih_l0.grad) == type(None):
            print("DecoderRNN IH weights are none")
        if type(self.rnn.weight_hh_l0.grad) == type(None):
            print("DecoderRNN HH weights are none")
        # else:
        #     print("DecoderRNN HH weight sum ", torch.sum(self.rnn.weight_hh_l0.data))

        return output, hidden

    def init_hidden_lstm(self):
        result = (Variable(torch.zeros(self.num_layers, 1, self.hidden_size)),
                  Variable(torch.zeros(self.num_layers, 1, self.hidden_size)))
        if USE_CUDA:
            return result.cuda()
        else:
            return result

    def init_hidden_gru(self):
        result = Variable(torch.zeros(self.num_layers, 1, self.hidden_size))
        if USE_CUDA:
            return result.cuda()
        else:
            return result


#
# ===============================================
# ===============================================
# ===============================================
#
# Dense Layers


class EncoderDense(nn.Module):
    def __init__(self, hidden_dim, z_dim):
        super(EncoderDense, self).__init__()
        self.hidden_dim = hidden_dim
        self.fc_mu = nn.Linear(hidden_dim, z_dim)
        self.fc_sig = nn.Linear(hidden_dim, z_dim)

    def forward(self, x):
        x = x.view(-1, self.hidden_dim)
        z_mu = self.fc_mu(x)
        z_sigma = torch.exp(self.fc_sig(x))

        if type(self.fc_mu.weight.grad) == type(None):
            print("EncoderDense mu grad is none")
        if type(self.fc_sig.weight.grad) == type(None):
            print("EncoderDense sig grad is none")

        return z_mu, z_sigma


class DecoderDense(nn.Module):
    def __init__(self, z_dim, hidden_dim):
        super(DecoderDense, self).__init__()
        self.hidden_dim = hidden_dim
        self.fc = nn.Linear(z_dim, hidden_dim)

    def forward(self, z):
        hidden = self.fc(z)
        
        if type(self.fc.weight.grad) == type(None):
            print("DecoderDense grad is none")
        # else:
        #     print("DecoderDense weight sum", torch.sum(self.fc.weight.data))

        return hidden


#
# ===============================================
# ===============================================
# ===============================================
#
# VRAE


class VRAE(nn.Module):
    # by default our latent space is 50-dimensional
    # and we use 400 hidden units
    def __init__(self,
                 dataset,
                 vocab_dim,
                 sentence_embedding_size,
                 outer_lstm_hidden_dim,
                 z_dim,
                 decoder_hidden_dim,
                 max_length,
                 num_layers_for_rnns,
                 use_cuda=False):
        super(VRAE, self).__init__()

        #fastText (for creating output phrases)
        self.ftext = fasttext.FastText()

        # define rnns
        self.num_layers = num_layers_for_rnns
        self.outer_lstm_hidden_dim = outer_lstm_hidden_dim
        #self.encoder_rnn = EncoderRNN(input_size=vocab_dim,
        #                              hidden_size=encoder_hidden_dim,
        #                              num_layers=self.num_layers,
        #                              ftext = self.ftext)
        
        self.outer_lstm_rnn = nn.GRU(sentence_embedding_size,
                                      outer_lstm_hidden_dim,
                                      num_layers=self.num_layers)

        self.decoder_rnn = DecoderRNN(hidden_size=decoder_hidden_dim,
                                      output_size=vocab_dim,
                                      num_layers=self.num_layers)

        # define dense modules
        self.encoder_dense = EncoderDense(hidden_dim=outer_lstm_hidden_dim,
                                          z_dim=z_dim)

        self.decoder_dense = DecoderDense(z_dim=z_dim,
                                          hidden_dim=decoder_hidden_dim)

        #self.context = []

        if use_cuda:
            # calling cuda() here will put all the parameters of
            # the encoder and decoder networks into gpu memory
            self.cuda()
        self.use_cuda = use_cuda
        self.z_dim = z_dim
        self.dataset = dataset
        self.max_length = max_length

    def bag_of_words_embedding(self, sentence_words):
        #This function is computed by VNRAE because it
        #already has a FastText object - more resource efficient
        #than instantiating a second FastText
        
        #print("bag_of_words_embedding")
        #print(sentence_words)
        
        sentence_vector = torch.zeros(300)

        #TODO: Should we ignore the 'SOS' and 'EOS' tokens,
        # or should we incorporate them into the embedding?
        for word in sentence_words:
            try:
                sentence_vector += self.ftext.get_vector(word)
            except:
                # word was not found in the model
                # probably because we truncated to 50000
                pass
        # divide by full number of words even if some of them
        # weren't found in the model. Why? Because it makes the
        # final vectors shorters, and thus (perhaps?) differentiates embeddings
        # with missing data from embeddings where we found all
        # the words.
        sentence_vector = sentence_vector/float(len(sentence_words))
        
        #print("sentence vector", sentence_vector)
        #input(">")

        return sentence_vector

    def model(self, input_variable, target_variable, step):
        # register PyTorch module `decoder` with Pyro
        pyro.module("decoder_dense", self.decoder_dense)
        pyro.module("decoder_rnn", self.decoder_rnn)

        # setup hyperparameters for prior p(z)
        # the type_as ensures we get CUDA Tensors if x is on gpu
        z_mu = ng_zeros([self.num_layers, self.z_dim], type_as=target_variable.data)
        z_sigma = ng_ones([self.num_layers, self.z_dim], type_as=target_variable.data)

        # sample from prior
        # (value will be sampled by guide when computing the ELBO)
        z = pyro.sample("latent", dist.normal, z_mu, z_sigma)

        # init vars
        target_length = target_variable.shape[0]

        decoder_input = dataset.to_onehot([[self.ftext.SOS_index]])
        decoder_input = decoder_input.cuda() if USE_CUDA else decoder_input

        decoder_outputs = np.ones((target_length))
        decoder_hidden = self.decoder_dense(z)

        for di in range(target_length):
            decoder_output, decoder_hidden = self.decoder_rnn(
                decoder_input, decoder_hidden)

            if self.use_cuda:
                decoder_outputs[di] = np.argmax(decoder_output.cpu().data.numpy())
            else:
                decoder_outputs[di] = np.argmax(decoder_output.data.numpy())
            
            if TEACHER_FORCING and step%100 != 0:
                #train using teacher forcing
                decoder_input = target_variable[di]
            else:
                #but sample without it
                val = self.dataset.to_onehot(np.array([decoder_outputs[di]]))
                decoder_input = val

            pyro.observe("obs_{}".format(di), dist.bernoulli, target_variable[di], decoder_output[0])

        # ----------------------------------------------------------------
        # prepare offer
        if self.use_cuda:
            offer = np.argmax(input_variable.cpu().data.numpy(), axis=1).astype(int)
        else:
            offer = np.argmax(input_variable.data.numpy(), axis=1).astype(int)

        # prepare answer
        if self.use_cuda:
            answer = np.argmax(target_variable.cpu().data.numpy(), axis=1).astype(int)
        else:
            answer = np.argmax(target_variable.data.numpy(), axis=1).astype(int)

        # prepare rnn
        rnn_response = list(map(int, decoder_outputs))
        
        # print output
        if step % 10 == 0:
            print("---------------------------")
            #print("Offer: ", ' '.join(self.ftext.get_words_from_indices(offer)))
            print("Input/Target:", ' '.join(self.ftext.get_words_from_indices(answer)))
            print("RNN:", ' '.join(self.ftext.get_words_from_indices(rnn_response)))
        # ----------------------------------------------------------------


    def guide(self, input_variable, target_variable, step):
        # register PyTorch module `encoder` with Pyro
        pyro.module("encoder_dense", self.encoder_dense)
        #pyro.module("encoder_rnn", self.encoder_rnn)
        pyro.module("outer_lstm_rnn", self.outer_lstm_rnn)

        # Collect the last n sentence embeddings
        #self.context.append(input_variable)
        #if len(self.context) > CONTEXT_LENGTH:
        #    self.context = self.context[1:]

        # OUTER LSTM
        # recurrently encode each of the sentence embeddings
            
        # init_vars
        outer_lstm_hidden = Variable(torch.zeros(self.num_layers, 1, self.outer_lstm_hidden_dim))
        outer_lstm_hidden = outer_lstm_hidden.cuda() if USE_CUDA else outer_lstm_hidden

        # loop to encode, final hidden state is the conversation embedding
        #for i in range(len(self.context)):


        #print("HERE - vnrae guide")
        #print(input_variable.shape)
        #print(len(input_variable))
        for i in range(len(input_variable)):
            #print(i)
            #print(input_variable[i])
            #print(input_variable[i].shape)
            #print(input_variable[i].view(-1,1,SENTENCE_EMBEDDING_SIZE).shape)
            #sys.stdout.flush()
            #outer_lstm_output, outer_lstm_hidden = self.outer_lstm_rnn(
            #    input_variable[i].view(1,1,-1), outer_lstm_hidden)
            outer_lstm_output, outer_lstm_hidden = self.outer_lstm_rnn(
                input_variable[i].view(-1,1,SENTENCE_EMBEDDING_SIZE), outer_lstm_hidden)
            #print('finished loop')

        # use the encoder to get the parameters used to define q(z|x)
        z_mu, z_sigma = self.encoder_dense(outer_lstm_hidden)

        # sample the latent code z
        pyro.sample("latent", dist.normal, z_mu, z_sigma)


#
# ===============================================
# ===============================================
# ===============================================
#
# Training loop


num_epochs = 100
test_frequency = 1

vrae = VRAE(dataset,
            VOCAB_SIZE,
            SENTENCE_EMBEDDING_SIZE,
            OUTER_LSTM_HIDDEN_SIZE,
            Z_DIMENSION,
            DECODER_HIDDEN_SIZE,
            MAX_LENGTH,
            NUM_LAYERS_FOR_RNNS,
            USE_CUDA)
optimizer = optim.Adam({"lr": LEARNING_RATE})
svi = SVI(vrae.model, vrae.guide, optimizer, loss="ELBO")

#print("SVI DIR", svi.__dir__())

f=fasttext.FastText()

total_training_steps = 0
losses=[]
for epoch in range(30):
    print("Start epoch!")
    # initialize loss accumulator
    epoch_loss = 0.
    # do a training epoch over each mini-batch x
    # returned by the data loader
    #context = torch.zeros(CONTEXT_LENGTH, SENTENCE_EMBEDDING_SIZE) 
    for convo_i in range(dataset.size()):
        total_training_steps += 1

        # the data loader gives us two sentences, but we
        # will discard the second and use only the first sentence
        next_sentence, _ = dataset.next_batch()

        # generate target one_hot outputs for the input sentence
        y = f.get_indices(next_sentence)
        y_onehot = dataset.to_onehot(y, long_type=False)

        # embed the input sentence
        embedded_x = vrae.bag_of_words_embedding(next_sentence)
        #if convo_i % 10 == 0:
        #    print("Next sentence:", ' '.join(next_sentence))
        
        #for i in range(len(context)-1):
        #    context[i] = context[i+1]
        #context[-1] = embedded_sentence
        #
        #x = context

        # do ELBO gradient and accumulate loss
        if USE_CUDA:
            loss = svi.step(embedded_x.cuda().view(1,SENTENCE_EMBEDDING_SIZE), y_onehot.cuda(), convo_i)
        else:
            loss = svi.step(embedded_x.view(1,SENTENCE_EMBEDDING_SIZE), y_onehot, convo_i)
        epoch_loss += loss

        # print loss
        if convo_i % 10 == 0:
            print("Epoch: {}, Step: {}, NLL: {}".format(epoch, convo_i, loss))
            print("---------------------------\n")

        if total_training_steps % 1000 == 0:
            losses.append(loss)
        if total_training_steps % 1000 == 0:
            # save the plot data
            pickle.dump(losses, open('data_decoder/losses_bag_of_words.pkl','wb'))
            #save the model
            pickle.dump(svi, open('data_decoder/model_bag_of_words.pkl', 'wb'))

    print("\n\nTrained epoch: {}, epoch loss: {}\n\n".format(epoch, epoch_loss))
