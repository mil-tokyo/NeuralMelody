#!/usr/bin/env python
# -*- coding: utf8 -*-

import json
import os
import argparse
import scipy.io as sio
import numpy as np
from hmmlearn.hmm import MultinomialHMM
import tensorflow as tf
from sklearn.utils import check_random_state

# import matplotlib.pylab as plt

def post_processing_parts(matrix, ratio):
    # Cast in int and repeat four time
    A = np.repeat(matrix, ratio)
    # Add bar index information which might be useful
    bar_counter = np.mod(np.arange(len(A)), np.zeros(len(A))+ratio)
    B = A * ratio + bar_counter
    return B.astype(int)


def build_proba(var, cond):
  # Count occurences
  dim = (int(np.max(var))+1, int(np.max(cond))+1)
  proba = np.zeros((dim))
  # Normalize
  for (v, c) in zip(var, cond):
    proba[int(v), int(c)] += 1
  # Normalize along var axis
  return np.transpose(proba / proba.sum(axis=0))


class MultinomialHMM_prod(MultinomialHMM):
  def __init__(self, n_components=1,
                 startprob_prior=1.0, transmat_prior=1.0,
                 algorithm="viterbi", random_state=None,
                 n_iter=10, tol=1e-2, verbose=False,
                 params="ste", init_params="ste"):
        MultinomialHMM.__init__(self, n_components=n_components,
                 startprob_prior=startprob_prior, transmat_prior=transmat_prior,
                 algorithm=algorithm, random_state=random_state,
                 n_iter=n_iter, tol=tol, verbose=verbose,
                 params=params, init_params=init_params)
        return

  def _generate_sample_from_state_PROD(self, state, cond_matrix, cond, random_state=None):
    cum_prod = np.cumsum(self.emissionprob_[state, :] * cond_matrix[cond, :])
    cdf = cum_prod / np.max(cum_prod)
    random_state = check_random_state(random_state)
    return [(cdf > random_state.rand()).argmax()]

  def sampling_prod_hmm(self, cond_matrix, cond_variable, random_state=None):
    n_samples = len(cond_variable)
    if random_state is None:
        random_state = self.random_state
    random_state = check_random_state(random_state)

    startprob_cdf = np.cumsum(self.startprob_)
    transmat_cdf = np.cumsum(self.transmat_, axis=1)

    currstate = (startprob_cdf > random_state.rand()).argmax()
    curr_cond = cond_variable[0]
    state_sequence = [currstate]
    
    X = [self._generate_sample_from_state_PROD(
        currstate, cond_matrix, curr_cond, random_state=random_state)]

    for t in range(n_samples - 1):
        currstate = (transmat_cdf[currstate] > random_state.rand()) \
            .argmax()
        curr_cond = cond_variable[t+1]
        state_sequence.append(currstate)
        X.append(self._generate_sample_from_state_PROD(
            currstate, cond_matrix, curr_cond, random_state=random_state))

    return np.atleast_2d(X), np.array(state_sequence, dtype=int)

def main(params):
  DEBUG = params['DEBUG']
  dataset = params['dataset']
  nh_part = params['nh_part']
  nh_chords = params['nh_chords']
  num_gen = params['num_gen']

  ##################################################################
  # DATA PROCESSING
  # Songs indices
  song_indices = [43,85,133,183,225,265,309,349,413,471,519,560,590,628,670,712,764,792,836,872,918,966,1018,1049,1091,1142,1174,1222,1266,1278,1304,1340,1372,1416,1456,1484,1536,1576,1632,1683,1707,1752,1805,1857,1891,1911]
  # Chords mapping
  chord_names = ['C;Em', 'A#;F', 'Dm;Em', 'Dm;G', 'Dm;C', 'Am;Em', 'F;C', 'F;G', 'Dm;F', 'C;C', 'C;E', 'Am;G', 'F;Em', 'F;F', 'G;G', 'Am;Am', 'Dm;Dm', 'C;A#', 'Em;F', 'C;G', 'G#;A#', 'F;Am', 'G#;Fm', 'Am;Gm', 'F;E', 'Dm;Am', 'Em;Em', 'G#;G#', 'Em;Am', 'C;Am', 'F;Dm', 'G#;G', 'F;A#', 'Am;G#', 'C;D', 'G;Am', 'Am;C', 'Am;A#', 'A#;G', 'Am;F', 'A#;Am', 'E;Am', 'Dm;E', 'A;G', 'Am;Dm', 'Em;Dm', 'C;F#m', 'Am;D', 'G#;Em', 'C;Dm', 'C;F', 'G;C', 'A#;A#', 'Am;Caug', 'Fm;G', 'A;A']

  # Import .mat file
  dataset_root = os.path.join('data', dataset)
  mat_path = os.path.join(dataset_root, 'augmented_futari.mat')
  data_mat = sio.loadmat(mat_path)
  chords_per_part = 2
  chords_per_bar = 4
  num_chords = 56
  num_parts = 4
  sub_sampling_ratio_parts = chords_per_bar/chords_per_part

  # Get parts
  parts_data_ = (np.dot(np.transpose(data_mat["feats"][-num_parts:]), np.asarray(range(num_parts))).astype(int)).reshape(-1, 1)
  # Group by bar
  parts_data = parts_data_[::sub_sampling_ratio_parts]
  # Parts with position in bar. Used condition chords generation
  parts_bar_data = post_processing_parts(parts_data, sub_sampling_ratio_parts)
  # Get chords transitions
  chords_data = (np.dot(np.transpose(data_mat["feats"][:-num_parts]), np.asarray(range(num_chords))).astype(int)).reshape(-1, 1)


  #################################
  # Group by song
  parts_length = []
  chords_length = []
  start_ind = 0
  for end_ind in song_indices:
    chords_length.append(end_ind - start_ind + 1)
    start_ind = end_ind + 1
  parts_length = [e/2 for e in chords_length]
  ##################################################################
  
  ##################################################################
  # PARTS
  # Compute HMM for part modeling
  hmm_part = MultinomialHMM(n_components=nh_part, n_iter=20)
  hmm_part.fit(parts_data, parts_length)

  # def plot_mat(matrix, name):
  #   fig = plt.figure()
  #   ax = fig.add_subplot(1,1,1)
  #   ax.set_aspect('equal')
  #   plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.ocean)
  #   plt.colorbar()
  #   plt.savefig(name, format='pdf')

  # plot_mat(hmm_part.transmat_, 'part_transmat.pdf')
  # plot_mat(np.reshape(hmm_part.startprob_, [-1, 1]), 'part_startprob.pdf')
  # plot_mat(hmm_part.emissionprob_, 'part_emissionprob.pdf')
  ##################################################################
  
  ##################################################################
  # CHORDS
  hmm_chords = MultinomialHMM_prod(n_components=nh_chords, n_iter=20)
  hmm_chords.fit(chords_data, chords_length)
  # plot_mat(hmm_chords.transmat_, 'chords_transmat.pdf')
  # plot_mat(np.reshape(hmm_chords.startprob_, [-1, 1]), 'chords_startprob.pdf')
  # plot_mat(hmm_chords.emissionprob_, 'chords_emissionprob.pdf')
  ##################################################################
  
  #################################
  # GENERATION
  # Sample sequence
  for n in range(num_gen):
    gen_part_sequence_, _ = hmm_part.sample(params["gen_seq_length"])
    gen_part_sequence = post_processing_parts(gen_part_sequence_, sub_sampling_ratio_parts)
    # Compute conditioning on parts
    p_chords_given_partBar = build_proba(chords_data, parts_bar_data)
    gen_chord_sequence, _ = hmm_chords.sampling_prod_hmm(p_chords_given_partBar, gen_part_sequence)
    ######## T E S T  ################
    # Independent HMM ?
    # gen_chord_sequence, _ = hmm_chords.sampling(n_samples=44)
    ##################################
    if params["DEBUG"]:
      with open("results_chords/" + str(n), 'wb') as f:
        for count, (part, chord) in enumerate(zip(gen_part_sequence, gen_chord_sequence)):
          if count % 2 == 0:
            f.write(str(part/2) + " ; " + chord_names[chord[0]] + "\n")
          else:
            f.write("  ; " + chord_names[chord[0]] + "\n")
          if count % 8 == 7:
            f.write("\n")
  gen_part_sequence = [e/2 for e in gen_part_sequence]
  return gen_part_sequence, gen_chord_sequence, num_chords, num_parts

if __name__ == "__main__":

  parser = argparse.ArgumentParser()

  # Data
  parser.add_argument('-d', '--dataset', dest='dataset', default='music', help='dataset: flickr8k/flickr30k')
  # Parts' HMM
  parser.add_argument('--nh_part', dest='nh_part', type=int, default=20, help='number of hidden states for the part\'s HMM')
  parser.add_argument('--nh_chords', dest='nh_chords', type=int, default=40, help='number of hidden states for the part\'s HMM')
  # Generation
  parser.add_argument('--gen_seq_length', type=int, default=8, help='length of the generated sequences')
  parser.add_argument('--num_gen', dest='num_gen', type=int, default=10, help='number sequences generated (i.e. sampling n times from the hmm)')
  parser.add_argument('--DEBUG', dest='DEBUG', type=bool, default=False, help='True = debug mode on')

  args = parser.parse_args()
  params = vars(args) # convert to ordinary dict
  print 'parsed parameters:'
  print json.dumps(params, indent=2)

  main(params)
