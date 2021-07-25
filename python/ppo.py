#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.tokenize import WordPunctTokenizer

import random
from tqdm import tqdm
from collections import Counter

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import scipy.signal
import time

pd.set_option('display.max_rows', 500)


# In[ ]:


from Stub.Stub import StubSession
from grammar_lib.testCaseModifier import GCheckModifier
from grammar_lib.SQLChecker import parser

# with open("grammar_lib/all_grammar_inputs.txt") as file:
#     all_grammar_inputs = [line.strip() for line in file]

all_grammar_inputs = ["' ) UNION select NULL,email,pass,NULL from user;",
                        "' ) UNION select NULL,email,pass,NULL from user;",
                        "' null UNION select NULL,email,pass,NULL from user;",
                        "' null UNION select NULL,email,pass,NULL from user;",
                        "' ) UNION select NULL,email,pass,NULL from user",
                        "' ) UNION select NULL,email,pass,NULL from user"]
    
all_grammar_inputs_mod = []
for gram_inp in all_grammar_inputs:
    all_grammar_inputs_mod.append(gram_inp)
    all_grammar_inputs_mod.append(gram_inp.replace("'", ')'))

all_grammar_inputs = all_grammar_inputs_mod.copy()


# In[ ]:


sent = all_grammar_inputs.copy()

sent_processed = []
tokenizer = WordPunctTokenizer()
max_len = 0

for input_str in sent:
    input_str = str(input_str)
    input_str = input_str.lower()
    input_str = input_str.strip()
    str_list = nltk.word_tokenize(input_str)
    if len(str_list) > max_len:
        max_len = len(str_list)
    sent_processed.append(str_list)
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

VOCAB_SIZE, max_len


# In[ ]:


"""
## Hyperparameters
"""

# Hyperparameters of the PPO algorithm
MODEL_NAME = "RLFuzzPPOSearchPen_v1"
steps_per_epoch = 50
epochs = 5000
gamma = 0.99
clip_ratio = 0.2
policy_learning_rate = 3e-4
value_function_learning_rate = 1e-3
train_policy_iterations = 80
train_value_iterations = 80
lam = 0.97
target_kl = 0.01
hidden_sizes = (64, 64)


# In[ ]:


MAX_LENGTH = 15 # including EOS

loaded_test = [] # store compatible length grammar based init (generation) test strings
for sent in sent_processed:
    if len(sent)<=MAX_LENGTH-1:
        loaded_test.append(sent)
        
len(loaded_test)


# In[ ]:


def eos_and_ind(sampled_list: list, ind: bool = False):
    if ind:
        eos_token = 0
    else:
        eos_token = "<EOS>"
    clip_ind = sampled_list.index(eos_token)
    # clip_ind = np.where(sampled_list==eos_token)[0][0] # for numpy
    if clip_ind < MAX_LENGTH-1:
        clip_ind_rem = MAX_LENGTH-clip_ind
        sampled_list = sampled_list[:clip_ind]+[eos_token]*clip_ind_rem # slower than if
    assert len(sampled_list) == MAX_LENGTH
    if ind:
        return sampled_list
    return [word2ind[s] for s in sampled_list]

def init_string_list():
    gram_gen_str = loaded_test[random.randint(0, len(loaded_test)-1)] # randint a<=N<=b
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


# In[ ]:


class RLFuzz:
    def __init__(self, rewarding: list = None):
        if rewarding is None:
            self.init_string = init_string_list() # always as index
        else:
            self.init_string = rewarding.copy()
        self.seed_str = self.init_string.copy()
        self.last_str = self.init_string.copy()

    def __str__(self):
        return f'\nOrig seed string: \n{" ".join([ind2word[s] for s in self.init_string])}\nLast seed string: \n{" ".join([ind2word[s] for s in self.last_str])}\nCurr seed string: \n{" ".join([ind2word[s] for s in self.seed_str])}'

    def action(self, pos, vocab):
        self.last_str = self.seed_str.copy()
        self.seed_str = mutate_string_list(seed=self.seed_str, pos=pos, vocab=vocab)

class RLFuzzEnv:
    EXCEPTION_PENALTY = 0.1
    MUTATION_PENALTY = 0.2
    SAME_STRING_PENALTY = 0.3 # eos & grammar related
    PARSER_PENALTY = 0.5
    SUCCESS_REWARD = 5
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
        
        self.operation = 'None'
        
        return np.array(observation)

    def step(self, action):
        action_pos, action_vocab = self.breakdown_action(action)
        fuzzing_success = False # init  
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
        
        if self.episode_step>1 and (len(username_rl.strip()) == 0 or last_username_rl==username_rl): # rudimentary
            # print(f"SAME_STRING_PENALTY @ {self.episode_step}: ", username_rl)
            reward = -self.SAME_STRING_PENALTY
        elif parser_failed: # parser
            # print(f"PARSER_PENALTY @ {self.episode_step}: ", username_rl)
            reward = -self.PARSER_PENALTY
        else: # check via website
            if self.last_status == 1:
                self.session.reset_session()

            url="http://localhost/demo/example_mysql_injection_search_box.php"
            jsonFilePath = './Stub/conditions1.json'
            receive=self.session.s.get(url)
            form_details,keys=self.session.preprocessing_Form_Fields(url)

            values=[username_rl]
            logindata=self.session.form_input_feeding(keys,values,form_details)
            pass_Conditions, fail_Conditions = self.session.jsonReading(jsonFilePath)
            status = self.session.validation(url, logindata, keys, pass_Conditions, fail_Conditions)
            self.last_status = status
            
            status_ex = self.session.exceptionCatcher(url, logindata)
            
            exception_success = True if status_ex==1 else False

            fuzzing_success = True if status==1 else False

            if fuzzing_success:
#                 print(f"\nSUCCESS_REWARD @ {self.episode_step}: ")
#                 print(self.fuzzer)
#                 print(f"pos: {action_pos}, vocab: {action_vocab} -> {ind2word[action_vocab]}")
                reward = self.SUCCESS_REWARD
            elif exception_success:
#                 print(f"EXCEPTION_REWARD @ {self.episode_step}: ", username_rl)
                reward = -self.EXCEPTION_PENALTY
            else:
                # print(f"MUTATION_PENALTY @ {self.episode_step}: ", username_rl)
                reward = -self.MUTATION_PENALTY

        done = False
        if self.episode_step >= steps_per_epoch or (fuzzing_success and self.episode_step>steps_per_epoch//4): #TODO: chk -- removed: @ SUCCESS_REWARD
            done = True

        return np.array(new_observation), reward, done
    
    def breakdown_action(self, action):
        action_pos = action//VOCAB_SIZE
        action_vocab = action%VOCAB_SIZE
        # print('POS: ', action_pos, 'VOCAB: ', action_vocab, 'CONV: ', action_pos*VOCAB_SIZE+action_vocab)
        return action_pos, action_vocab

    def squeeze_actions(self, action_pos, action_vocab):
        return action_pos*VOCAB_SIZE+action_vocab


# In[ ]:


def discounted_cumulative_sums(x, discount):
    # Discounted cumulative sums of vectors for computing rewards-to-go and advantage estimates
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class Buffer:
    # Buffer for storing trajectories
    def __init__(self, observation_dimensions, size, gamma=0.99, lam=0.95):
        # Buffer initialization
        self.observation_buffer = np.zeros(
            (size, observation_dimensions), dtype=np.float32
        )
        self.action_buffer = np.zeros(size, dtype=np.int32)
        self.advantage_buffer = np.zeros(size, dtype=np.float32)
        self.reward_buffer = np.zeros(size, dtype=np.float32)
        self.return_buffer = np.zeros(size, dtype=np.float32)
        self.value_buffer = np.zeros(size, dtype=np.float32)
        self.logprobability_buffer = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.pointer, self.trajectory_start_index = 0, 0

    def store(self, observation, action, reward, value, logprobability):
        # Append one step of agent-environment interaction
        self.observation_buffer[self.pointer] = observation
        self.action_buffer[self.pointer] = action
        self.reward_buffer[self.pointer] = reward
        self.value_buffer[self.pointer] = value
        self.logprobability_buffer[self.pointer] = logprobability
        self.pointer += 1

    def finish_trajectory(self, last_value=0):
        # Finish the trajectory by computing advantage estimates and rewards-to-go
        path_slice = slice(self.trajectory_start_index, self.pointer)
        rewards = np.append(self.reward_buffer[path_slice], last_value)
        values = np.append(self.value_buffer[path_slice], last_value)

        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]

        self.advantage_buffer[path_slice] = discounted_cumulative_sums(
            deltas, self.gamma * self.lam
        )
        self.return_buffer[path_slice] = discounted_cumulative_sums(
            rewards, self.gamma
        )[:-1]

        self.trajectory_start_index = self.pointer

    def get(self):
        # Get all data of the buffer and normalize the advantages
        self.pointer, self.trajectory_start_index = 0, 0
        advantage_mean, advantage_std = (
            np.mean(self.advantage_buffer),
            np.std(self.advantage_buffer),
        )
        self.advantage_buffer = (self.advantage_buffer - advantage_mean) / advantage_std
        return (
            self.observation_buffer,
            self.action_buffer,
            self.advantage_buffer,
            self.return_buffer,
            self.logprobability_buffer,
        )


def mlp(x, sizes, activation=tf.tanh, output_activation=None):
    # Build a feedforward neural network
    x = layers.Embedding(VOCAB_SIZE, 16)(x)
    x= layers.GRU(32, return_sequences=False)(x)
    x = layers.Dropout(0.2)(x)
    for size in sizes[:-1]:
        x = layers.Dense(units=size, activation=activation)(x)
    return layers.Dense(units=sizes[-1], activation=output_activation)(x)


def logprobabilities(logits, a):
    # Compute the log-probabilities of taking actions a by using the logits (i.e. the output of the actor)
    logprobabilities_all = tf.nn.log_softmax(logits)
    logprobability = tf.reduce_sum(
        tf.one_hot(a, num_actions) * logprobabilities_all, axis=1
    )
    return logprobability


# Sample action from actor
@tf.function
def sample_action(observation):
    logits = actor(observation)
    action = tf.squeeze(tf.random.categorical(logits, 1), axis=1)
    return logits, action


# Train the policy by maxizing the PPO-Clip objective
@tf.function
def train_policy(
    observation_buffer, action_buffer, logprobability_buffer, advantage_buffer
):

    with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
        ratio = tf.exp(
            logprobabilities(actor(observation_buffer), action_buffer)
            - logprobability_buffer
        )
        min_advantage = tf.where(
            advantage_buffer > 0,
            (1 + clip_ratio) * advantage_buffer,
            (1 - clip_ratio) * advantage_buffer,
        )

        policy_loss = -tf.reduce_mean(
            tf.minimum(ratio * advantage_buffer, min_advantage)
        )
    policy_grads = tape.gradient(policy_loss, actor.trainable_variables)
    policy_optimizer.apply_gradients(zip(policy_grads, actor.trainable_variables))

    kl = tf.reduce_mean(
        logprobability_buffer
        - logprobabilities(actor(observation_buffer), action_buffer)
    )
    kl = tf.reduce_sum(kl)
    return kl


# Train the value function by regression on mean-squared error
@tf.function
def train_value_function(observation_buffer, return_buffer):
    with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
        value_loss = tf.reduce_mean((return_buffer - critic(observation_buffer)) ** 2)
    value_grads = tape.gradient(value_loss, critic.trainable_variables)
    value_optimizer.apply_gradients(zip(value_grads, critic.trainable_variables))


# In[ ]:


"""
## Initializations
"""
random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)

agg_rewards = []
# Initialize the environment and get the dimensionality of the
# observation space and the number of possible actions
env = RLFuzzEnv()
observation_dimensions = MAX_LENGTH
num_actions = MAX_LENGTH*VOCAB_SIZE
print('num_actions: ', num_actions, f'{MAX_LENGTH}*{VOCAB_SIZE}')

# Initialize the buffer
buffer = Buffer(observation_dimensions, steps_per_epoch)

# Initialize the actor and the critic as keras models
observation_input = keras.Input(shape=(observation_dimensions,), dtype=tf.float32)
logits = mlp(observation_input, list(hidden_sizes) + [num_actions], tf.tanh, None)
actor = keras.Model(inputs=observation_input, outputs=logits)
value = tf.squeeze(
    mlp(observation_input, list(hidden_sizes) + [1], tf.tanh, None), axis=1
)
critic = keras.Model(inputs=observation_input, outputs=value)

# Initialize the policy and the value function optimizers
policy_optimizer = keras.optimizers.Adam(learning_rate=policy_learning_rate)
value_optimizer = keras.optimizers.Adam(learning_rate=value_function_learning_rate)

# Initialize the observation, episode return and episode length
observation, episode_return, episode_length = env.reset(), 0, 0

"""
## Train
"""
# Iterate over the number of epochs
for epoch in tqdm(range(epochs), ascii=True, unit='episodes'):
    # Initialize the sum of the returns, lengths and number of episodes for each epoch
    sum_return = 0
    sum_length = 0
    num_episodes = 0
    
    success_count = 0

    # Iterate over the steps of each epoch
    for t in range(steps_per_epoch):
        # Get the logits, action, and take one step in the environment
        observation = observation.reshape(1, -1)
        logits, action = sample_action(observation)
        
        obs_text_list = [ind2word[i] for i in observation[0]]
        if ";" not in obs_text_list and obs_text_list.index("<EOS>")!=MAX_LENGTH-1 and random.random()<=0.9:
            # print("--- semicolon ---@", t)
            action = [env.squeeze_actions(action_pos = obs_text_list.index("<EOS>"), action_vocab = word2ind[";"])]
            action = tf.constant(action, dtype=tf.int64)
        
        elif "'" in obs_text_list and ")" not in obs_text_list and random.random()<=0.9:
            # print("--- ') ---@", t)
            action = [env.squeeze_actions(action_pos = obs_text_list.index("'")+1, action_vocab = word2ind[")"])]
            action = tf.constant(action, dtype=tf.int64)
            
        elif "'" not in obs_text_list and random.random()<=0.9:
            # print("--- apos --- @", t)
            action = [env.squeeze_actions(action_pos = 0, action_vocab = word2ind["'"])]
            action = tf.constant(action, dtype=tf.int64)
            
        observation_new, reward, done = env.step(action[0].numpy())
        if reward > 0: success_count+=1
#         print(" ".join(obs_text_list))
#         print(" ".join(ind2word[i] for i in observation_new))
#         print()
        episode_return += reward
        episode_length += 1

        # Get the value and log-probability of the action
        value_t = critic(observation)
        logprobability_t = logprobabilities(logits, action)

        # Store obs, act, rew, v_t, logp_pi_t
        buffer.store(observation, action, reward, value_t, logprobability_t)

        # Update the observation
        observation = observation_new

        # Finish trajectory if reached to a terminal state
        terminal = done
        if terminal or (t == steps_per_epoch - 1):
            last_value = 0 if done else critic(observation.reshape(1, -1))
            buffer.finish_trajectory(last_value)
            sum_return += episode_return
            sum_length += episode_length
            num_episodes += 1
            observation, episode_return, episode_length = env.reset(), 0, 0

    # Get values from the buffer
    (
        observation_buffer,
        action_buffer,
        advantage_buffer,
        return_buffer,
        logprobability_buffer,
    ) = buffer.get()

    # Update the policy and implement early stopping using KL divergence
    for _ in range(train_policy_iterations):
        kl = train_policy(
            observation_buffer, action_buffer, logprobability_buffer, advantage_buffer
        )
        if kl > 1.5 * target_kl:
            # Early Stopping
            break

    # Update the value function
    for _ in range(train_value_iterations):
        train_value_function(observation_buffer, return_buffer)

    # Print mean return and length for each epoch
    agg_rewards.append(round((sum_return / num_episodes), 2))
    print(
        f" Epoch: {epoch + 1}. Mean Return: {(sum_return / num_episodes):.2f}. Mean Length: {(sum_length / num_episodes):.2f}. Success Count: {success_count}"
    )
    
    if epoch%100==0:
        mean_agg = sum(agg_rewards[-100:])/len(agg_rewards[-100:])
        print("Mean Aggregate: ", mean_agg)
        actor.save(f'models/{MODEL_NAME}__ep_{epoch}__{(sum_return / num_episodes):.2f}__actor.h5')
        critic.save(f'models/{MODEL_NAME}__ep_{epoch}__{(sum_return / num_episodes):.2f}__critic.h5')


# In[ ]:


actor.save(f'models/{MODEL_NAME}_FINAL__ep_{epoch}__{(sum_return / num_episodes):.2f}__actor.h5')
critic.save(f'models/{MODEL_NAME}_FINAL__ep_{epoch}__{(sum_return / num_episodes):.2f}__critic.h5')


# In[ ]:




