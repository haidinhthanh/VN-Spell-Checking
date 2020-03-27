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
from tensorflow.python.ops.rnn_cell_impl import _zero_state_tensors
import time
import re
from sklearn.model_selection import train_test_split
# from google.colab import drive
# drive.mount('/content/drive')

filename=['sentence.csv','scrapy.csv','essay.csv','test.csv']

sentences=[]
for file in filename:
	with open(file, 'r', encoding="utf8") as f:
	  reader = csv.reader(f)
	  your_list = list(reader)
	  sentences1=[_ for i in range(len(your_list)) for _ in your_list[i]]
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
print(sorted(vocab_to_int))

f=open("vocab_to_int.txt","w+", encoding="utf8")
for character, value in vocab_to_int.items():
	f.write(character+"equal"+str(value)+"\n")
f.close()

int_to_vocab = {}
for character, value in vocab_to_int.items():
    int_to_vocab[value] = character
# print(int_to_vocab)

f=open("int_to_vocab.txt","w+", encoding="utf8")
for  value,character in int_to_vocab.items():
	f.write(str(value)+"equal"+character+"\n")
f.close()

# Convert sentences to integers
int_sentences = []

for clean_sentence in clean_sentences:
    int_sentence = []
    for character in clean_sentence:
        int_sentence.append(vocab_to_int[character])
    int_sentences.append(int_sentence)

# Limit the data we will use to train our model
max_length = 150
min_length = 6

good_sentences = []

for int_sentence in int_sentences:
    if len(int_sentence) <= max_length and len(int_sentence) >= min_length:
        good_sentences.append(int_sentence)

print("We will use {} to train and test our model.".format(len(good_sentences)))

# Split the data into training and testing sentences
training, testing = train_test_split(good_sentences, test_size = 0.15, random_state = 2)

print("Number of training sentences:", len(training))
print("Number of testing sentences:", len(testing))

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

def process_encoding_input(targets, vocab_to_int, batch_size):
    '''Remove the last word id from each batch and concat the <GO> to the begining of each batch'''

    with tf.name_scope("process_encoding"):
        ending = tf.strided_slice(targets, [0, 0], [batch_size, -1], [1, 1])
        dec_input = tf.concat([tf.fill([batch_size, 1], vocab_to_int['<GO>']), ending], 1)

    return dec_input

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

def pad_sentence_batch(sentence_batch):
    """Pad sentences with <PAD> so that each sentence of a batch has the same length"""
    max_sentence = max([len(sentence) for sentence in sentence_batch])
    return [sentence + [vocab_to_int['<PAD>']] * (max_sentence - len(sentence)) for sentence in sentence_batch]


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

# The default parameters
epochs = 32
batch_size = 128
num_layers = 2
rnn_size = 512
embedding_size = 128
learning_rate = 0.0005
direction = 2
threshold = 0.95
keep_probability = 0.75

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


# ## Training the Model

def train(model, epochs, log_string):
    '''Train the RNN'''

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Used to determine when to stop the training early
        testing_loss_summary = []

        # Keep track of which batch iteration is being trained
        iteration = 0

        display_step = 30 # The progress of the training will be displayed after every 30 batches
        stop_early = 0
        stop = 5 # If the batch_loss_testing does not decrease in 5 consecutive checks, stop training
        per_epoch = 1 # Test the model 3 times per epoch
        testing_check = (len(training_sorted)//batch_size//per_epoch)-1

        print()
        print("Training Model: {}".format(log_string))

        train_writer = tf.summary.FileWriter('./logs/1/train/{}'.format(log_string), sess.graph)
        test_writer = tf.summary.FileWriter('./logs/1/test/{}'.format(log_string))

        for epoch_i in range(1, epochs+1):
            batch_loss = 0
            batch_time = 0

            for batch_i, (input_batch, target_batch, input_length, target_length) in enumerate(
                    get_batches(training_sorted, batch_size, threshold)):
                start_time = time.time()

                summary, loss, _ = sess.run([model.merged,
                                             model.cost,
                                             model.train_op],
                                             {model.inputs: input_batch,
                                              model.targets: target_batch,
                                              model.inputs_length: input_length,
                                              model.targets_length: target_length,
                                              model.keep_prob: keep_probability})


                batch_loss += loss
                end_time = time.time()
                batch_time += end_time - start_time

                # Record the progress of training
                train_writer.add_summary(summary, iteration)

                iteration += 1

                if batch_i % display_step == 0 and batch_i > 0:
                    print('Epoch {:>3}/{} Batch {:>4}/{} - Loss: {:>6.3f}, Seconds: {:>4.2f}'
                          .format(epoch_i,
                                  epochs,
                                  batch_i,
                                  len(training_sorted) // batch_size,
                                  batch_loss / display_step,
                                  batch_time))
                    batch_loss = 0
                    batch_time = 0

                #### Testing ####
                if batch_i % testing_check == 0 and batch_i > 0:
                    batch_loss_testing = 0
                    batch_time_testing = 0
                    for batch_i, (input_batch, target_batch, input_length, target_length) in enumerate(
                            get_batches(testing_sorted, batch_size, threshold)):
                        start_time_testing = time.time()
                        summary, loss = sess.run([model.merged,
                                                  model.cost],
                                                     {model.inputs: input_batch,
                                                      model.targets: target_batch,
                                                      model.inputs_length: input_length,
                                                      model.targets_length: target_length,
                                                      model.keep_prob: 1})

                        batch_loss_testing += loss
                        end_time_testing = time.time()
                        batch_time_testing += end_time_testing - start_time_testing

                        # Record the progress of testing
                        test_writer.add_summary(summary, iteration)

                    n_batches_testing = batch_i + 1
                    print('Testing Loss: {:>6.3f}, Seconds: {:>4.2f}'
                          .format(batch_loss_testing / n_batches_testing,
                                  batch_time_testing))

                    batch_time_testing = 0

                    # If the batch_loss_testing is at a new minimum, save the model
                    testing_loss_summary.append(batch_loss_testing)
                    if batch_loss_testing <= min(testing_loss_summary):
                        print('New Record!')
                        stop_early = 0
                        checkpoint = "./{}.ckpt".format(log_string)
                        saver = tf.train.Saver()
                        saver.save(sess, checkpoint)

                    else:
                        print("No Improvement.")
                        stop_early += 1
                        if stop_early == stop:
                            break

            if stop_early == stop:
                print("Stopping Training.")
                break

# Train the model with the desired tuning parameters
for keep_probability in [0.75]:
    for num_layers in [2]:
        for threshold in [0.95]:
            log_string = 'kp={},nl={},th={}'.format(keep_probability,
                                                    num_layers,
                                                    threshold)
            model = build_graph(keep_probability, rnn_size, num_layers, batch_size,
                                learning_rate, embedding_size, direction)
            train(model, epochs, log_string)
	
	
# ???????????????????????????????????????????????????
import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.layers import Embedding, GRU, Layer, Dense
import pandas as pd
from constant import DATA_PATH
import os
import time
from os import path
import re
from asset.number import NUMBER
from asset.letter import VN_LETTER
from asset.vowel import LiST_S_TONES
from asset.punctuation_mark import PUNCTUATION_MARK

# load data
df = pd.read_excel(path.join(DATA_PATH, 'train_spell.xlsx'), sheet_name="data train")
codes = ['<PAD>', '<GO>', '<EOS>', '<UNK>']


# create data set
def pre_processing_sentence(sentence):
    sentence = re.sub(r'([.,!?()])', r' \1 ', str(sentence))
    sentence = re.sub(r'\s{2,}', ' ', sentence)
    sentence = re.sub(r'\<|\>', '', sentence)
    sentence = '<GO>' + sentence + '<EOS>'
    return sentence


err_sentences = list()
cor_sentences = list()
for index, row in df.iterrows():
    err_sentence = pre_processing_sentence(df.loc[index]['error sentences'])
    cor_sentence = pre_processing_sentence(df.loc[index]['correct sentences'])
    err_sentences.append(err_sentence)
    cor_sentences.append(cor_sentence)


#
#
# def tokenize(sentences):
#     sentence_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters=" ")
#     sentence_tokenizer.fit_on_texts(sentences)
#     tensor = sentence_tokenizer.texts_to_sequences(sentences)
#     tensor = tf.keras.preprocessing.sequence.pad_sequences(sequences=tensor,
#                                                            padding='post')
#     return tensor, sentence_tokenizer
#
#
# input_tensor_train, inp_lang = tokenize(err_sentences)
# out_tensor_train, out_lang = tokenize(cor_sentences)
#
# BUFFER_SIZE = len(input_tensor_train)
# BATCH_SIZE = 64
# steps_per_epoch = len(input_tensor_train)//BATCH_SIZE
# embedding_dim = 256
# lstm_unit = 1024
# VOCAB_SIZE = len(inp_lang.word_index)+1
# vocab_tar_size = len(out_lang.word_index)+1
# print(out_lang.word_index['<GO>'])
#
# dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, out_tensor_train)).shuffle(BUFFER_SIZE)
# dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
# inp, out = next(iter(dataset))
def init_vocabulary(sentences):
    vocab_word_to_int = dict()
    vocab_int_to_word = dict()
    count = 0

    for letter_code in codes:
        vocab_word_to_int[letter_code] = count
        count += 1
    for number in NUMBER:
        vocab_word_to_int[number] = count
        count += 1
    for letter in VN_LETTER:
        vocab_word_to_int[letter] = count
        count += 1
    for vowel in LiST_S_TONES:
        vocab_word_to_int[vowel] = count
        count += 1
    for mark in PUNCTUATION_MARK:
        vocab_word_to_int[mark] = count
        count += 1
    vocab_word_to_int[" "] = count
    for key, value in vocab_word_to_int.items():
        vocab_int_to_word[value] = key

    return vocab_word_to_int, vocab_int_to_word


vocab_word_to_int_, vocab_int_to_word_ = init_vocabulary(sentences=err_sentences + cor_sentences)


def convert_sentences_to_int(sentences, vocab):
    convert_sen = list()
    sentences = [str(sentence).replace("<GO>", "").replace("<EOS>", "").strip()
                 for sentence in sentences]
    for sentence in sentences:
        new_sentence = list()
        for letter in sentence:
            if letter in vocab.keys():
                new_sentence.append(vocab[letter])
            else:
                new_sentence.append(vocab['<UNK>'])
        new_sentence.insert(0, vocab['<GO>'])
        new_sentence.append(vocab['<EOS>'])
        convert_sen.append(new_sentence)
    return convert_sen


int_err_sentences = convert_sentences_to_int(err_sentences, vocab_word_to_int_)
int_cor_sentences = convert_sentences_to_int(cor_sentences, vocab_word_to_int_)

BUFFER_SIZE = len(int_err_sentences)
BATCH_SIZE = 64
STEP_PER_EPOCH = len(int_err_sentences) // BATCH_SIZE
VOCAB_SIZE = max([value for value in vocab_word_to_int_.values()]) + 1
max_input_sen_len = max([len(sen) for sen in int_err_sentences])
max_out_sen_len = max([len(sen) for sen in int_cor_sentences])
dropout_ = 0.8
embedding_dim = 256
units = 1024
EPOCHS = 1


#
class Encoder(Model):
    def __init__(self, vocab_size, embedding_size, enc_units, batch_size, dropout):
        super(Encoder, self).__init__()
        self.embedding = Embedding(input_dim=vocab_size,
                                   output_dim=embedding_size)
        self.ec_units = enc_units
        self.batch_size = batch_size

        self.gru = GRU(enc_units,
                       return_sequences=True,
                       return_state=True,
                       recurrent_initializer='glorot_uniform',
                       dropout=dropout)

    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state=hidden)
        return output, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_size, self.ec_units))


#
#
# #
# # exam = int_err_sentences[0:64]
#
#
# #
# # exam = pad_sentence_batch(exam, max_input_sen_len)
# # exam = np.array(exam, dtype="int")
# # exam = tf.convert_to_tensor(value=exam)
# # hidden_state = encoder.initialize_hidden_state()
# # output_enc, enc_state = encoder.call(exam, hidden_state)
#
#
class BahdanauAttention(Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        query_with_time_axis = tf.expand_dims(query, 1)

        score = self.V(tf.nn.tanh(self.W1(query_with_time_axis) + self.W2(values)))
        attention_weights = tf.nn.softmax(score, axis=1)

        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


class Decoder(Model):
    def __init__(self, vocab_size, embedding_size, dec_units, batch_size, dropout):
        super(Decoder, self).__init__()
        self.batch_sz = batch_size
        self.dec_units = dec_units
        self.embedding = Embedding(vocab_size, embedding_size)
        self.gru = GRU(self.dec_units,
                       return_sequences=True,
                       return_state=True,
                       recurrent_initializer='glorot_uniform',
                       dropout=dropout)
        self.fc = Dense(vocab_size)
        self.attention = BahdanauAttention(self.dec_units)

    def call(self, x, hidden, enc_output):
        context_vector, attention_weights = self.attention(hidden, enc_output)
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        output, state = self.gru(x)
        output = tf.reshape(output, (-1, output.shape[2]))
        x = self.fc(output)

        return x, state, attention_weights


encoder = Encoder(vocab_size=VOCAB_SIZE,
                  embedding_size=embedding_dim,
                  enc_units=units,
                  batch_size=BATCH_SIZE,
                  dropout=dropout_)
decoder = Decoder(vocab_size=VOCAB_SIZE,
                  embedding_size=embedding_dim,
                  dec_units=units,
                  batch_size=BATCH_SIZE,
                  dropout=dropout_)

optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                            reduction='none')


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


checkpoint_dir = './checkpoint'
checkpoint_prefix = path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)


def train_step(inp, targ, enc_hidden):
    loss = 0

    with tf.GradientTape() as tape:
        enc_output, enc_hidden = encoder(inp, enc_hidden)

        dec_hidden = enc_hidden

        dec_input = tf.expand_dims([vocab_word_to_int_['<GO>']] * BATCH_SIZE, 1)
        for t in range(1, targ.shape[1]):
            predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)

            loss += loss_function(targ[:, t], predictions)

            dec_input = tf.expand_dims(targ[:, t], 1)

    batch_loss = (loss / int(targ.shape[1]))

    variables = encoder.trainable_variables + decoder.trainable_variables

    gradients = tape.gradient(loss, variables)

    optimizer.apply_gradients(zip(gradients, variables))

    return batch_loss


def pad_sentence_batch(sentence_batch, max_sentence):
    return [sentence + [vocab_word_to_int_['<PAD>']] * (max_sentence - len(sentence)) for sentence in sentence_batch]


def get_batches(error_sentences, correct_sentences, batch_size):
    for batch_i in range(0, len(error_sentences)//batch_size):
        start_i = batch_i * batch_size
        err_sentences_batch = error_sentences[start_i:start_i + batch_size]
        cor_sentences_batch = correct_sentences[start_i:start_i + batch_size]
        pad_err_sentences_batch = np.array(pad_sentence_batch(err_sentences_batch, max_input_sen_len))
        pad_cor_sentences_batch = np.array(pad_sentence_batch(cor_sentences_batch, max_out_sen_len))

        yield pad_err_sentences_batch, pad_cor_sentences_batch


for epoch in range(EPOCHS):
    start = time.time()

    enc_hidden = encoder.initialize_hidden_state()
    total_loss = 0

    for (batch, (inp, targ)) in enumerate(get_batches(int_err_sentences, int_cor_sentences, BATCH_SIZE)):
        batch_loss = train_step(inp, targ, enc_hidden)
        total_loss += batch_loss

        if batch % 100 == 0:
            print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                         batch,
                                                         batch_loss.numpy()))
    # saving (checkpoint) the model every 2 epochs
    if (epoch + 1) % 2 == 0:
        checkpoint.save(file_prefix=checkpoint_prefix)

    print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                        total_loss / STEP_PER_EPOCH))
    print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
# /////////////////////////////////////////////////////////////////
NUMBER = "0 1 2 3 4 5 6 7 8 9".split()
"""
dau ngat quang cau
"""
PUNCTUATION_MARK = [
    ".",
    "?",
    "...",
    ":",
    "!",
    "-",
    "(",
    ")",
    '"',
    ";",
    ",",
    "[",
    "]",
    "+",
    "-",
    "*",
    "/",
    "#",
    "$",
    "%",
    "^",
    "&",
    "{",
    "}",
    "=",
    "@"
]
TONE_S_A = "a à á ả ã ạ ă ằ ắ ẳ ẵ ặ â ầ ấ ẩ ẫ ậ".split()
TONE_S_E = "e è é ẻ ẽ ẹ ê ề ế ể ễ ệ".split()
TONE_S_I = "i ì í ỉ ĩ ị".split()
TONE_S_O = "o ò ó ỏ õ ọ ô ồ ố ổ ỗ ộ ơ ờ ớ ở ỡ ợ".split()
TONE_S_U = "u ù ú ủ ũ ụ ư ừ ứ ử ữ ự".split()
TONE_S_Y = "y ỳ ý ỷ ỹ ỵ".split()

LiST_S_TONES = TONE_S_A \
               + TONE_S_E \
               + TONE_S_I \
               + TONE_S_O \
               + TONE_S_U \
               + TONE_S_Y

VN_LETTER = "a ă â b c d đ e ê g h i k l m n o ô ơ p q r s t u ư v x y".split()
