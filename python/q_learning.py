#!/usr/bin/env python
# coding: utf-8

# In[1]:

import pandas as pd
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.tokenize import WordPunctTokenizer

import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten, Input, Lambda
from tensorflow.keras.layers import  RepeatVector, TimeDistributed, GlobalMaxPooling1D, Embedding, Permute
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow import keras
from collections import deque
import time
import random
from tqdm import tqdm
import os
from collections import Counter

pd.set_option('display.max_rows', 500)


# In[2]:


# Grammar1 to Grammar5 files stored in all_grammar_inputs.txt


# In[3]:


from Stub.Stub import StubSession


# In[4]:


from grammar_lib.testCaseModifier import GCheckModifier
from grammar_lib.SQLChecker import parser


# In[5]:


with open("grammar_lib/all_grammar_inputs.txt") as file:
    all_grammar_inputs = [line.strip() for line in file]


# In[6]:


sent = all_grammar_inputs.copy()

sent_processed = []
tokenizer = WordPunctTokenizer()

for input_str in sent:
    input_str = str(input_str)
    input_str = input_str.lower()
    input_str = input_str.strip()
    sent_processed.append(nltk.word_tokenize(input_str))
    # sent_processed.append(tokenizer.tokenize(input_str))
    
all_vocab = [v_w for v in sent_processed for v_w in v]

v_count = dict(Counter(all_vocab))
v_count = dict(sorted(v_count.items(), key=lambda item: item[1], reverse=True))
all_vocab = list(v_count.keys())

all_vocab.sort()

ind_list = list(range(1, len(all_vocab)+1))

word2ind = dict(zip(all_vocab, ind_list))
ind2word = dict(zip(ind_list, all_vocab))

word2ind['<EOS>'] = 0
ind2word[0] = '<EOS>'

all_vocab = ['<EOS>']+all_vocab

VOCAB_SIZE = len(all_vocab)

VOCAB_SIZE


# In[7]:


# gmod = GCheckModifier()
# gparse = parser()

# str_test = "('  UNION select NULL email from DUAL #"#all_grammar_inputs[1000]
# print(str_test)
# for_parsing = gmod.grammarchecker(str_test) 
# print(for_parsing)
# gparse.main(for_parsing) # 1 for failed


# In[8]:


DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 5_000  # last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 150 
MINIBATCH_SIZE = 128
UPDATE_TARGET_EVERY = 5
MODEL_NAME = 'RLFuzzv0.1'
MIN_REWARD = -200
MAX_LENGTH = 11 # including EOS

EPISODES = 1_000

epsilon = 1
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001

AGGREGATE_STATS_EVERY = 100  # episodes

# succ = [word2ind[s] for s in "( ' OR 1 = 1 ; -- )".lower().split()]+[0, 0] # maintain max_length & EOS


# In[9]:


loaded_test = [] # store compatible length grammar based init (generation) test strings
for sent in sent_processed:
    if len(sent)<=MAX_LENGTH-1:
        loaded_test.append(sent)


# In[10]:


def eos_and_ind(sampled_list: list, ind: bool = False):
    if ind:
        eos_token = 0
    else:
        eos_token = "<EOS>"
    clip_ind = sampled_list.index(eos_token)
    if clip_ind < MAX_LENGTH-1:
        clip_ind_rem = MAX_LENGTH-clip_ind
        sampled_list = sampled_list[:clip_ind]+[eos_token]*clip_ind_rem # slower than if
    assert len(sampled_list) == MAX_LENGTH
    if ind:
        return sampled_list
    return [word2ind[s] for s in sampled_list]

def init_string_list():
    gram_gen_str = loaded_test[random.randint(0, len(loaded_test))]
    sampled_list = gram_gen_str+(MAX_LENGTH-len(gram_gen_str))*['<EOS>']
    # sampled_list = list(random.sample(all_vocab, MAX_LENGTH-1))+['<EOS>']
    return eos_and_ind(sampled_list)

def mutate_string_list(seed: list, pos: int = None, vocab: int = None):
    if vocab is None:
        vocab_str = random.sample(all_vocab, 1)[0]
        vocab = word2ind[vocab_str]
    if pos is None:
        pos = random.randint(0, MAX_LENGTH-2) # MAX_LENGTH-2 inclusive
    if pos != MAX_LENGTH-1: # should never replace EOS
        seed[pos] = vocab
    return eos_and_ind(seed, ind=True)


# In[19]:


class RLFuzz:
    def __init__(self, rewarding=None):
        if rewarding is None:
            self.init_string = init_string_list() # always as index
        else:
            self.init_string = rewarding.copy()
        self.seed_str = self.init_string.copy()
        self.last_str = self.init_string.copy()

    def __str__(self):
        return f'\nOrg: \n{" ".join([ind2word[s] for s in self.init_string])}\n Current seed string: \n{" ".join([ind2word[s] for s in self.seed_str])}'

    def action(self, pos, vocab):
        self.last_str = self.seed_str.copy()
        self.seed_str = mutate_string_list(seed=self.seed_str, pos=pos, vocab=vocab)

class RLFuzzEnv:
    MUTATION_PENALTY = 1
    SAME_STRING_PENALTY = 10 # eos & grammar related
    PARSER_PENALTY = 20
    SUCCESS_REWARD = 1000
    ACTION_SPACE_SIZE_POS = MAX_LENGTH
    ACTION_SPACE_SIZE_VOCAB = VOCAB_SIZE

    def reset(self, rewarding=None):
        self.session = StubSession()
        self.last_status = 0
        self.fuzzer = RLFuzz(rewarding)
        self.episode_step = 0
        observation = self.fuzzer.init_string
        
        self.gmod = GCheckModifier()
        self.gparse = parser()
        
        return observation

    def step(self, action_pos, action_vocab):       
        self.episode_step += 1
        self.fuzzer.action(action_pos, action_vocab)
        new_observation = self.fuzzer.seed_str
        
        eos_index = new_observation.index(0)
        new_observation_ = new_observation[:eos_index]
        username_rl = " ".join([ind2word[s] for s in new_observation_])
                
        eos_indexL = self.fuzzer.last_str.index(0)
        new_observationL_ = self.fuzzer.last_str[:eos_indexL]
        last_username_rl = " ".join([ind2word[s] for s in new_observationL_])
                
        parser_failed = self.gparse.main(self.gmod.grammarchecker(username_rl))
        
        if len(username_rl.strip()) == 0 or last_username_rl==username_rl: # rudimentary
#             print(f"SAME_STRING_PENALTY @ {self.episode_step}: ", username_rl)
            reward = -self.SAME_STRING_PENALTY
        elif parser_failed: # parser
#             print(f"PARSER_PENALTY @ {self.episode_step}: ", username_rl)
            reward = -self.PARSER_PENALTY
        else: # check via website
            if self.last_status == 1:
                self.session.reset_session()

            url="http://localhost/demo/example_mysql_injection_login.php"
            jsonFilePath = './Stub/conditions.json'
            receive=self.session.s.get(url)
            form_details,keys=self.session.preprocessing_Form_Fields(url)

            values=[username_rl, "RaNdOmStRiNg"]
            logindata=self.session.form_input_feeding(keys,values,form_details)
            pass_Conditions, fail_Conditions = self.session.jsonReading(jsonFilePath)
            status = self.session.validation(url, logindata, keys, pass_Conditions, fail_Conditions)
            self.last_status = status

            fuzzing_success = True if status==1 else False

            if fuzzing_success:
                print(f"SUCCESS_REWARD @ {self.episode_step}: ", username_rl)
                reward = self.SUCCESS_REWARD
            else:
    #             print(f"MUTATION_PENALTY @ {self.episode_step}: ", username_rl)
                reward = -self.MUTATION_PENALTY

        done = False
        if self.episode_step >= 100: #TODO: chk -- removed: @ SUCCESS_REWARD
            done = True

        return new_observation, reward, done

# Agent class
class DQNAgent:
    def __init__(self):

        self.model = self.create_model()
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        self.target_update_counter = 0

        #self.tensorboard = ModifiedTensorBoard(log_dir="logs\\{}-{}".format(MODEL_NAME, int(time.time())), profile_batch = 10000000)
        #$REM
        
    def create_model(self):
        inputs = Input(shape=(MAX_LENGTH,))

        embed=Embedding(VOCAB_SIZE, 100)(inputs)

        activations= keras.layers.GRU(250, return_sequences=True)(embed)

        attention = TimeDistributed(Dense(1, activation='tanh'))(activations)
        attention = Flatten()(attention)
        attention = Activation('softmax')(attention)
        attention = RepeatVector(250)(attention)
        attention = Permute([2, 1])(attention)

        sent_representation = keras.layers.multiply([activations, attention])
        sent_representation = Lambda(lambda xin: K.sum(xin, axis=1))(sent_representation)

        x_pos = Dense(32, activation="relu")(sent_representation)
        x_pos = Dropout(0.1)(x_pos)
        x_pos = Dense(env.ACTION_SPACE_SIZE_POS, activation='linear', name='q_pos')(x_pos)

        x_vocab = Dense(32, activation="relu")(sent_representation)
        x_vocab = Dropout(0.1)(x_vocab)
        x_vocab = Dense(env.ACTION_SPACE_SIZE_VOCAB, activation='linear', name='q_vocab')(x_vocab)

        model = keras.Model(inputs=inputs, outputs=[x_pos, x_vocab])
        model.compile(loss=["mse", "mse"], optimizer=Adam(lr=0.001), metrics=["accuracy", "accuracy"])
        return model

    # (observation space, action_pos, action_vocab, reward, new observation space, done)
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def train(self, terminal_state, step):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        # minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        # Get current states from minibatch, then query NN model for Q values
        current_states = np.array([transition[0] for transition in minibatch])
        current_qs_list = self.model.predict(current_states)
        # print(current_qs_list[0].shape, current_qs_list[1].shape) # 64,11 & 64,60

        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        new_current_states = np.array([transition[-2] for transition in minibatch])
        future_qs_list = self.target_model.predict(new_current_states)

        X, y_pos, y_vocab = [], [], []

        # Now we need to enumerate our batches
        for index, (current_state, action_pos, action_vocab, reward, new_current_state, done) in enumerate(minibatch):

            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here
            if not done:
                max_future_q_pos = np.max(future_qs_list[0][index])
                new_q_pos = reward + DISCOUNT * max_future_q_pos
                
                max_future_q_vocab = np.max(future_qs_list[1][index])
                new_q_vocab = reward + DISCOUNT * max_future_q_vocab
            else:
                new_q_pos = reward
                new_q_vocab = reward

            # Update Q value for given state
            current_qs_pos = current_qs_list[0][index]
            current_qs_pos[action_pos] = new_q_pos
            
            current_qs_vocab = current_qs_list[1][index]
            current_qs_vocab[action_vocab] = new_q_vocab

            # And append to our training data
            X.append(current_state)
            y_pos.append(current_qs_pos)
            y_vocab.append(current_qs_vocab)

        # Fit on all samples as one batch, log only on terminal state
        self.model.fit(np.array(X), [y_pos, y_vocab], batch_size=MINIBATCH_SIZE, verbose=0, 
                       shuffle=False if terminal_state else None)
        #$REM: ,callbacks=[self.tensorboard] 

        # Update target network counter every episode
        if terminal_state:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    # Queries main network for Q values given current observation space (environment state)
    def get_qs(self, state):
        return self.model.predict(np.expand_dims(np.array(state), axis=0))


# In[20]:


env = RLFuzzEnv()
ep_rewards = [-200]

random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)

if not os.path.isdir('models'):
    os.makedirs('models')

agent = DQNAgent()

for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):

    # Update tensorboard step every episode #$REM
    # agent.tensorboard.step = episode

    # Restarting episode - reset episode reward, step number, and env + get current state
    episode_reward = 0
    step = 1
#     if episode == 2:
#         current_state = env.reset(succ)
#     else:
    current_state = env.reset()

    done = False
    while not done:
        if np.random.random() > epsilon:
            q_values = agent.get_qs(current_state)
            action_pos = np.argmax(q_values[0][0])
            action_vocab = np.argmax(q_values[1][0])
            # action = np.argmax(agent.get_qs(current_state))
        else:
            action_pos = np.random.randint(0, env.ACTION_SPACE_SIZE_POS)
            action_vocab = np.random.randint(0, env.ACTION_SPACE_SIZE_VOCAB)

        new_state, reward, done = env.step(action_pos, action_vocab)
        episode_reward += reward

        # At every step update replay memory and train main network
        agent.update_replay_memory((current_state, action_pos, action_vocab, reward, new_state, done))
        agent.train(done, step)

        current_state = new_state
        step += 1

    # Logging
    ep_rewards.append(episode_reward)
    if not episode % AGGREGATE_STATS_EVERY or episode == 1:
        average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
        min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
        max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
        
        #$REM agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)

        if min_reward > MIN_REWARD:
            agent.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

    # Decay epsilon
    if epsilon > MIN_EPSILON:
        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)


# In[ ]:





# In[ ]:


# class ModifiedTensorBoard(TensorBoard):

#     # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.step = 1
#         self.writer = tf.summary.create_file_writer(self.log_dir)
#         self._log_write_dir = self.log_dir

#     # Overriding this method to stop creating default log writer
#     def set_model(self, model):
#         pass

#     # Overrided, saves logs with our step number
#     # (otherwise every .fit() will start writing from 0th step)
#     def on_epoch_end(self, epoch, logs=None):
#         self.update_stats(**logs)

#     # Overrided
#     # We train for one batch only, no need to save anything at epoch end
#     def on_batch_end(self, batch, logs=None):
#         pass

#     # Overrided, so won't close writer
#     def on_train_end(self, _):
#         pass

#     # Custom method for saving own metrics
#     # Creates writer, writes custom metrics and closes writer
#     def update_stats(self, **stats):
#         self._write_logs(stats, self.step)
        
#     def _write_logs(self, logs, index):
#         with self.writer.as_default():
#             for name, value in logs.items():
#                 tf.summary.scalar(name, value, step=index)
#                 self.step += 1
#                 self.writer.flush()


# In[ ]:


# class TransformerBlock(layers.Layer):
#     def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
#         super(TransformerBlock, self).__init__()
#         self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
#         self.ffn = keras.Sequential(
#             [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
#         )
#         self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
#         self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
#         self.dropout1 = layers.Dropout(rate)
#         self.dropout2 = layers.Dropout(rate)

#     def call(self, inputs, training):
#         attn_output = self.att(inputs, inputs)
#         attn_output1 = self.dropout1(attn_output, training=training)
#         out1 = self.layernorm1(inputs + attn_output1)
#         ffn_output = self.ffn(out1)
#         ffn_output = self.dropout2(ffn_output, training=training)
#         return self.layernorm2(out1 + ffn_output), attn_output
    
# class TokenAndPositionEmbedding(layers.Layer):
#     def __init__(self, maxlen, vocab_size, embed_dim):
#         super(TokenAndPositionEmbedding, self).__init__()
#         self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
#         self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

#     def call(self, x):
#         maxlen = tf.shape(x)[-1]
#         positions = tf.range(start=0, limit=maxlen, delta=1)
#         positions = self.pos_emb(positions)
#         x = self.token_emb(x)
#         return x + positions

# embed_dim = 32  # Embedding size for each token
# num_heads = 2  # Number of attention heads
# ff_dim = 32  # Hidden layer size in feed forward network inside transformer

# inputs = Input(shape=(MAX_LENGTH,))
# embedding_layer = TokenAndPositionEmbedding(MAX_LENGTH, VOCAB_SIZE, embed_dim)
# x = embedding_layer(inputs)
# transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
# x, attn_output = transformer_block(x)
# x = GlobalAveragePooling1D()(x)
# x = Dropout(0.1)(x)
# x = Dense(64, activation="relu")(x)
# x = Dropout(0.1)(x)

# x_pos = Dense(32, activation="relu")(x)
# x_pos = Dropout(0.1)(x_pos)
# x_pos = Dense(env.ACTION_SPACE_SIZE_POS, activation='linear')(x_pos)

# x_vocab = Dense(32, activation="relu")(x)
# x_vocab = Dropout(0.1)(x_vocab)
# x_vocab = Dense(env.ACTION_SPACE_SIZE_VOCAB, activation='linear')(x_vocab)

# model = keras.Model(inputs=inputs, outputs=[x_pos, x_vocab])

# model.summary()

