import random as rd
import tensorflow as tf
import numpy as np
import csv
import pandas as pd
import os
from os import listdir
from os.path import isfile, join
from collections import namedtuple
from tensorflow.python.layers.core import Dense
import time
import re
from sklearn.model_selection import train_test_split
# from google.colab import drive

# drive.mount('/content/drive')
filename=['sentence.csv','scrapy.csv','test.csv']
sentences=[]
for file in filename:
    with open(file, 'r', encoding="utf8") as f:
      reader = csv.reader(f)
      your_list = list(reader)
      sentences1=[_ for i in range(len(your_list)) for _ in your_list[i]]
      print(len(sentences1))
      sentences+=sentences1


def clean_text(text):
    '''Remove unwanted characters and extra spaces from the text'''
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'[{}@_*>()\\#%+=\[\]]','', text)
    text = re.sub('a0','', text)
    text = re.sub('\'92t','\'t', text)
    text = re.sub('\'92s','\'s', text)
    text = re.sub('\'92m','\'m', text)
    text = re.sub('\'92ll','\'ll', text)
    text = re.sub('\'91','', text)
    text = re.sub('\'92','', text)
    text = re.sub('\'93','', text)
    text = re.sub('\'94','', text)
    text = re.sub('\.','. ', text)
    text = re.sub('\!','! ', text)
    text = re.sub('\?','? ', text)
    text = re.sub(' +',' ', text)
    text = text.lower()
    return text
clean_sentences = []
for sentence in sentences:
    clean_sentences.append(clean_text(sentence)+".")
# print(clean_sentences[:5])

vocab_to_int = {}
count = 0
for clean_sentence in clean_sentences:
    for character in clean_sentence:
        if character not in vocab_to_int:
            vocab_to_int[character] = count
            count += 1

# Add special tokens to vocab_to_int
codes = ['<PAD>','<EOS>','<GO>']
for code in codes:
    vocab_to_int[code] = count
    count += 1
vocab_size = len(vocab_to_int)
# print("The vocabulary contains {} characters.".format(vocab_size))

# Convert sentences to integers
int_sentences = []

for clean_sentence in clean_sentences:
    int_sentence = []
    for character in clean_sentence:
        int_sentence.append(vocab_to_int[character])
    int_sentences.append(int_sentence)

# Limit the data we will use to train our model
max_length = 150
min_length = 10

good_sentences = []

for int_sentence in int_sentences:
    if len(int_sentence) <= max_length and len(int_sentence) >= min_length:
        good_sentences.append(int_sentence)

# Split the data into training and testing sentences
training, testing = train_test_split(good_sentences, test_size = 0.15, random_state = 2)
training_sorted = []
testing_sorted = []

for i in range(min_length, max_length+1):
    for sentence in training:
        if len(sentence) == i:
            training_sorted.append(sentence)
    for sentence in testing:
        if len(sentence) == i:
            testing_sorted.append(sentence)

def removeNonAplhabet(inputlist):
    return [w for w in inputlist if w.isalpha()]
letter_VN=removeNonAplhabet(vocab_to_int)
# print(sorted(letter_VN))


CLASS_A = "a à á ả ã ạ ă ằ ắ ẳ ẵ ặ â ầ ấ ẩ ẫ ậ".split()
CLASS_E = "e è é ẻ ẽ ẹ ê ề ế ể ễ ệ".split()
CLASS_I = "i ì í ỉ ĩ ị".split()
CLASS_O = "o ò ó ỏ õ ọ ô ồ ố ổ ỗ ộ ơ ờ ớ ở ỡ ợ".split()
CLASS_U = "u ù ú ủ ũ ụ ư ừ ứ ử ữ ự".split()
CLASS_Y = "y ỳ ý ỷ ỹ ỵ".split()
CLASS_D = "d đ d đ d đ".split()

# 6 tones in total, including "unmarked" mark
# ngang = 0, huyền = 1, sắc = 2, hỏi = 3, ngã = 4, nặng = 5

NUM_TONES = 6
_ALL_CLASSES = [CLASS_A, CLASS_E, CLASS_I, CLASS_O, CLASS_U, CLASS_Y, CLASS_D]

def find_tone(syllable):
    for c in syllable:
        # found a vowel
        for cls in _ALL_CLASSES:
            # it is a toned vowel
            if c in cls[1:]:
                return cls.index(c) % NUM_TONES
    # unmarked
    return -1


def change_tone(syllable, tone_type):
    if tone_type > NUM_TONES:
        return syllable
    result = []
    changed = False
    for c in syllable:
        if changed:
            result.append(c)
            continue
        # found a vowel
        for cls in _ALL_CLASSES:
            # it is a toned vowel
            if c in cls[0:]:
                changed = True
                c = cls[cls.index(c) - (cls.index(c) % NUM_TONES) + tone_type]
                break
        result.append(c)
    return ''.join(result)

def noise_maker(sentence, threshold):
    '''Relocate, remove, or add characters to create spelling mistakes'''

    noisy_sentence = []
    i = 0
    while i < len(sentence):
        random = np.random.uniform(0,1,1)
        # Most characters will be correct since the threshold value is high
        if random < threshold:
            noisy_sentence.append(sentence[i])
        else:
            new_random = np.random.uniform(0,1,1)
            # ~25% chance characters will swap locations
            if new_random > 0.75:
                if i == (len(sentence) - 1):
                    # If last character in sentence, it will not be typed
                    continue
                else:
                    # if any other character, swap order with following character
                    noisy_sentence.append(sentence[i+1])
                    noisy_sentence.append(sentence[i])
                    i += 1
            # ~25% chance wrong tone
            elif new_random> 0.5:
                tone=find_tone(int_to_vocab[sentence[i]])
                if(tone==-1):
                    noisy_sentence.append(sentence[i])
                else:
                    newtone=rd.randrange(0,6,1)
                    while newtone == tone:
                        newtone=rd.randrange(0,6,1)
                    noisy_sentence.append(vocab_to_int[change_tone(int_to_vocab[sentence[i]],newtone)])
            # ~25% chance an extra lower case letter will be added to the sentence
            elif new_random >0.25:
                random_letter = np.random.choice(letter_VN, 1)[0]
                noisy_sentence.append(vocab_to_int[random_letter])
                noisy_sentence.append(sentence[i])
            # ~25% chance a character will not be typed
            else:
                pass
        i += 1
    return noisy_sentence

def text_to_ints(text):
    '''Prepare the text for the model'''

    text = clean_text(text)
    return [vocab_to_int[word] for word in text]

# print('  Response Words: {}'.format("".join([int_to_vocab[i] for i in noise_maker(text_to_ints("kia là một con mèo"),0.95)])))

def model_inputs():
    '''Create palceholders for inputs to the model'''

    with tf.name_scope('inputs'):
        inputs = tf.placeholder(tf.int32, [None, None], name='inputs')
    with tf.name_scope('targets'):
        targets = tf.placeholder(tf.int32, [None, None], name='targets')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    inputs_length = tf.placeholder(tf.int32, (None,), name='inputs_length')
    targets_length = tf.placeholder(tf.int32, (None,), name='targets_length')
    max_target_length = tf.reduce_max(targets_length, name='max_target_len')

    return inputs, targets, keep_prob, inputs_length, targets_length, max_target_length


# In[64]:

def process_encoding_input(targets, vocab_to_int, batch_size):
    '''Remove the last word id from each batch and concat the <GO> to the begining of each batch'''

    with tf.name_scope("process_encoding"):
        ending = tf.strided_slice(targets, [0, 0], [batch_size, -1], [1, 1])
        dec_input = tf.concat([tf.fill([batch_size, 1], vocab_to_int['<GO>']), ending], 1)

    return dec_input


# In[65]:

def encoding_layer(rnn_size, sequence_length, num_layers, rnn_inputs, keep_prob, direction):
    '''Create the encoding layer'''

    if direction == 1:
        with tf.name_scope("RNN_Encoder_Cell_1D"):
            for layer in range(num_layers):
                with tf.variable_scope('encoder_{}'.format(layer)):
                    lstm = tf.contrib.rnn.LSTMCell(rnn_size)

                    drop = tf.contrib.rnn.DropoutWrapper(lstm,
                                                         input_keep_prob = keep_prob)

                    enc_output, enc_state = tf.nn.dynamic_rnn(drop,
                                                              rnn_inputs,
                                                              sequence_length,
                                                              dtype=tf.float32)

            return enc_output, enc_state


    if direction == 2:
        with tf.name_scope("RNN_Encoder_Cell_2D"):
            for layer in range(num_layers):
                with tf.variable_scope('encoder_{}'.format(layer)):
                    cell_fw = tf.contrib.rnn.LSTMCell(rnn_size)
                    cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw,
                                                            input_keep_prob = keep_prob)

                    cell_bw = tf.contrib.rnn.LSTMCell(rnn_size)
                    cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw,
                                                            input_keep_prob = keep_prob)

                    enc_output, enc_state = tf.nn.bidirectional_dynamic_rnn(cell_fw,
                                                                            cell_bw,
                                                                            rnn_inputs,
                                                                            sequence_length,
                                                                            dtype=tf.float32)
            # Join outputs since we are using a bidirectional RNN
            enc_output = tf.concat(enc_output,2)
            # Use only the forward state because the model can't use both states at once
            return enc_output, enc_state[0]


# In[66]:

def training_decoding_layer(dec_embed_input, targets_length, dec_cell, initial_state, output_layer,
                            vocab_size, max_target_length):
    '''Create the training logits'''

    with tf.name_scope("Training_Decoder"):
        training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=dec_embed_input,
                                                            sequence_length=targets_length,
                                                            time_major=False)

        training_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell,
                                                           training_helper,
                                                           initial_state,
                                                           output_layer)

        training_logits, _, __ = tf.contrib.seq2seq.dynamic_decode(training_decoder,
                                                               output_time_major=False,
                                                               impute_finished=True,
                                                               maximum_iterations=max_target_length)
        return training_logits


# In[67]:

def inference_decoding_layer(embeddings, start_token, end_token, dec_cell, initial_state, output_layer,
                             max_target_length, batch_size):
    '''Create the inference logits'''

    with tf.name_scope("Inference_Decoder"):
        start_tokens = tf.tile(tf.constant([start_token], dtype=tf.int32), [batch_size], name='start_tokens')

        inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embeddings,
                                                                    start_tokens,
                                                                    end_token)

        inference_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell,
                                                            inference_helper,
                                                            initial_state,
                                                            output_layer)

        inference_logits, _, __ = tf.contrib.seq2seq.dynamic_decode(inference_decoder,
                                                                output_time_major=False,
                                                                impute_finished=True,
                                                                maximum_iterations=max_target_length)

        return inference_logits


# In[68]:

def decoding_layer(dec_embed_input, embeddings, enc_output, enc_state, vocab_size, inputs_length, targets_length,
                   max_target_length, rnn_size, vocab_to_int, keep_prob, batch_size, num_layers, direction):
    '''Create the decoding cell and attention for the training and inference decoding layers'''

    with tf.name_scope("RNN_Decoder_Cell"):
        for layer in range(num_layers):
            with tf.variable_scope('decoder_{}'.format(layer)):
                lstm = tf.contrib.rnn.LSTMCell(rnn_size)
                dec_cell = tf.contrib.rnn.DropoutWrapper(lstm,
                                                         input_keep_prob = keep_prob)

    output_layer = Dense(vocab_size,
                         kernel_initializer = tf.truncated_normal_initializer(mean = 0.0, stddev=0.1))

    attn_mech = tf.contrib.seq2seq.BahdanauAttention(rnn_size,
                                                  enc_output,
                                                  inputs_length,
                                                  normalize=False,
                                                  name='BahdanauAttention')

    with tf.name_scope("Attention_Wrapper"):
        dec_cell = tf.contrib.seq2seq.AttentionWrapper(dec_cell,attn_mech,rnn_size)

    initial_state =  dec_cell.zero_state(batch_size=batch_size,dtype=tf.float32).clone(cell_state=enc_state)


    with tf.variable_scope("decode"):
        training_logits = training_decoding_layer(dec_embed_input,
                                                  targets_length,
                                                  dec_cell,
                                                  initial_state,
                                                  output_layer,
                                                  vocab_size,
                                                  max_target_length)
    with tf.variable_scope("decode", reuse=True):
        inference_logits = inference_decoding_layer(embeddings,
                                                    vocab_to_int['<GO>'],
                                                    vocab_to_int['<EOS>'],
                                                    dec_cell,
                                                    initial_state,
                                                    output_layer,
                                                    max_target_length,
                                                    batch_size)

    return training_logits, inference_logits


# In[69]:

def seq2seq_model(inputs, targets, keep_prob, inputs_length, targets_length, max_target_length,
                  vocab_size, rnn_size, num_layers, vocab_to_int, batch_size, embedding_size, direction):
    '''Use the previous functions to create the training and inference logits'''

    enc_embeddings = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1, 1))
    enc_embed_input = tf.nn.embedding_lookup(enc_embeddings, inputs)
    enc_output, enc_state = encoding_layer(rnn_size, inputs_length, num_layers,
                                           enc_embed_input, keep_prob, direction)

    dec_embeddings = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1, 1))
    dec_input = process_encoding_input(targets, vocab_to_int, batch_size)
    dec_embed_input = tf.nn.embedding_lookup(dec_embeddings, dec_input)

    training_logits, inference_logits  = decoding_layer(dec_embed_input,
                                                        dec_embeddings,
                                                        enc_output,
                                                        enc_state,
                                                        vocab_size,
                                                        inputs_length,
                                                        targets_length,
                                                        max_target_length,
                                                        rnn_size,
                                                        vocab_to_int,
                                                        keep_prob,
                                                        batch_size,
                                                        num_layers,
                                                        direction)

    return training_logits, inference_logits


# In[70]:

def pad_sentence_batch(sentence_batch):
    """Pad sentences with <PAD> so that each sentence of a batch has the same length"""
    max_sentence = max([len(sentence) for sentence in sentence_batch])
    return [sentence + [vocab_to_int['<PAD>']] * (max_sentence - len(sentence)) for sentence in sentence_batch]


# In[71]:

def get_batches(sentences, batch_size, threshold):
    """Batch sentences, noisy sentences, and the lengths of their sentences together.
       With each epoch, sentences will receive new mistakes"""

    for batch_i in range(0, len(sentences)//batch_size):
        start_i = batch_i * batch_size
        sentences_batch = sentences[start_i:start_i + batch_size]

        sentences_batch_noisy = []
        for sentence in sentences_batch:
            sentences_batch_noisy.append(noise_maker(sentence, threshold))

        sentences_batch_eos = []
        for sentence in sentences_batch:
            sentence.append(vocab_to_int['<EOS>'])
            sentences_batch_eos.append(sentence)

        pad_sentences_batch = np.array(pad_sentence_batch(sentences_batch_eos))
        pad_sentences_noisy_batch = np.array(pad_sentence_batch(sentences_batch_noisy))

        # Need the lengths for the _lengths parameters
        pad_sentences_lengths = []
        for sentence in pad_sentences_batch:
            pad_sentences_lengths.append(len(sentence))

        pad_sentences_noisy_lengths = []
        for sentence in pad_sentences_noisy_batch:
            pad_sentences_noisy_lengths.append(len(sentence))

        yield pad_sentences_noisy_batch, pad_sentences_batch, pad_sentences_noisy_lengths, pad_sentences_lengths


# *Note: This set of values achieved the best results.*

# In[72]:

# The default parameters
epochs = 64
batch_size = 128
num_layers = 2
rnn_size = 512
embedding_size = 128
learning_rate = 0.0005
direction = 2
threshold = 0.95
keep_probability = 0.75


# In[73]:

def build_graph(keep_prob, rnn_size, num_layers, batch_size, learning_rate, embedding_size, direction):

    tf.reset_default_graph()

    # Load the model inputs
    inputs, targets, keep_prob, inputs_length, targets_length, max_target_length = model_inputs()

    # Create the training and inference logits
    training_logits, inference_logits = seq2seq_model(tf.reverse(inputs, [-1]),
                                                      targets,
                                                      keep_prob,
                                                      inputs_length,
                                                      targets_length,
                                                      max_target_length,
                                                      len(vocab_to_int)+1,
                                                      rnn_size,
                                                      num_layers,
                                                      vocab_to_int,
                                                      batch_size,
                                                      embedding_size,
                                                      direction)

    # Create tensors for the training logits and inference logits
    training_logits = tf.identity(training_logits.rnn_output, 'logits')

    with tf.name_scope('predictions'):
        predictions = tf.identity(inference_logits.sample_id, name='predictions')
        tf.summary.histogram('predictions', predictions)

    # Create the weights for sequence_loss
    masks = tf.sequence_mask(targets_length, max_target_length, dtype=tf.float32, name='masks')

    with tf.name_scope("cost"):
        # Loss function
        cost = tf.contrib.seq2seq.sequence_loss(training_logits,
                                                targets,
                                                masks)
        tf.summary.scalar('cost', cost)

    with tf.name_scope("optimze"):
        optimizer = tf.train.AdamOptimizer(learning_rate)

        # Gradient Clipping
        gradients = optimizer.compute_gradients(cost)
        capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
        train_op = optimizer.apply_gradients(capped_gradients)

    # Merge all of the summaries
    merged = tf.summary.merge_all()

    # Export the nodes
    export_nodes = ['inputs', 'targets', 'keep_prob', 'cost', 'inputs_length', 'targets_length',
                    'predictions', 'merged', 'train_op','optimizer']
    Graph = namedtuple('Graph', export_nodes)
    local_dict = locals()
    graph = Graph(*[local_dict[each] for each in export_nodes])

    return graph


vocab_to_int={}
f = open('vocab_to_int.txt', 'r')
for line in f.readlines():
    name,score = line.split("equal")
    vocab_to_int[name] = int(score)

int_to_vocab={}

f = open('int_to_vocab.txt', 'r')
for line in f.readlines():
    score,name = line.split("equal")
    int_to_vocab[int(score)] = name.replace("\n","")

def text_to_ints(text):
    '''Prepare the text for the model'''

    text = clean_text(text)
    return [vocab_to_int[word] for word in text]

# In[176]:

# Create your own sentence or use one from the dataset
text = "Dạ nà ra sao rồi"
text = text_to_ints(text)

#random = np.random.randint(0,len(testing_sorted))
#text = testing_sorted[random]
#text = noise_maker(text, 0.95)

checkpoint = "/content/drive/My Drive/Colab Notebooks/kp=0.75,nl=2,th=0.95.ckpt"
model = build_graph(keep_probability, rnn_size, num_layers, batch_size, learning_rate, embedding_size, direction)

with tf.Session() as sess:
    # Load saved model
    saver = tf.train.Saver()
    saver.restore(sess, checkpoint)

    #Multiply by batch_size to match the model's input parameters
    answer_logits = sess.run(model.predictions, {model.inputs: [text]*batch_size,
                                                 model.inputs_length: [len(text)]*batch_size,
                                                 model.targets_length: [len(text)+1],
                                                 model.keep_prob: [1.0]})[0]

# Remove the padding from the generated sentence
pad = vocab_to_int["<PAD>"]

print('\nText')
print('  Word Ids:    {}'.format([i for i in text]))
print('  Input Words: {}'.format("".join([int_to_vocab[i] for i in text])))

print('\nSummary')
print('  Word Ids:       {}'.format([i for i in answer_logits if i != pad]))
print('  Response Words: {}'.format("".join([int_to_vocab[i] for i in answer_logits if i != pad])))

# https://www.coursera.org/learn/machine-learning-with-python
# https://www.coursera.org/learn/introduction-to-deep-learning-with-keras?specialization=ai-engineer
# https://www.coursera.org/learn/deep-neural-networks-with-pytorch?specialization=ai-engineer
# https://www.coursera.org/learn/neural-networks-deep-learning?specialization=deep-learning
# https://www.coursera.org/learn/machine-learning-projects?specialization=deep-learning
# https://www.coursera.org/learn/convolutional-neural-networks?specialization=deep-learning
# https://www.coursera.org/learn/nlp-sequence-models
