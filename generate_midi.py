import argparse
import json
import time
import datetime
import numpy as np
import code
import socket
import os
import cPickle as pickle
import math
import pretty_midi
import scipy.io
import re

from imagernn.data_provider import getDataProvider
from imagernn.solver import Solver
from imagernn.imagernn_utils import decodeGenerator, eval_split

import chord_sequence_generation

program_num = 80
base_tempo = 120
new_track = pretty_midi.Instrument(program_num,is_drum=False,name='melody')
bass_track = pretty_midi.Instrument(38,is_drum=False,name='bass')

def convert_pos(p,c):
  if '&' not in p:
    return float(p)+float(2*c)
  nat,frac = p.split('&')
  nu, de = frac.split('/')
  nat = int(nat)
  nat += 2*c
  return float(nat)+(float(nu) / float(de))

def convert_dur(d):
  if '/' not in d:
    return float(d)
  nu,de = d.split('/')
  return float(nu)/float(de)

def two_hot_encoding(parts, chords, num_chords, num_parts):
  T = len(parts)
  output = np.zeros((T, num_chords+num_parts))
  for index, (part, chord) in enumerate(zip(parts, chords)):
    output[index, num_chords + part] = 1
    output[index, chord] = 1
  return output

def adjust_tempo(new_midi_data):
  bpm = new_midi_data.get_tempo_changes()[-1][0]
  min_length = 60. / (bpm * 4)

  for instrument in new_midi_data.instruments:
    for note in instrument.notes:
      note.start *= base_tempo / bpm
      note.end *= base_tempo / bpm

def quantize(new_midi_data):
  bpm = new_midi_data.get_tempo_changes()[-1][0]
  min_length = 60. / (bpm * 4)

  for instrument in new_midi_data.instruments:
    for note in instrument.notes:
      note.start = round(note.start / min_length) * min_length
      note.end = round(note.end / min_length) * min_length
      if note.end - note.start == 0:
        note.end += min_length

def gen_from_scratch(params):
  # load the checkpoint
  checkpoint_path = params['checkpoint_path']
  max_images = params['max_images']
  fout = params['output_file']
  tempo = params['tempo']

  print 'loading checkpoint %s' % (checkpoint_path, )
  checkpoint = pickle.load(open(checkpoint_path, 'rb'))
  checkpoint_params = checkpoint['params']
  dataset = checkpoint_params['dataset']
  params['dataset'] = dataset
  model = checkpoint['model']
  dump_folder = params['dump_folder']
  ixtoword = checkpoint['ixtoword']

  if dump_folder:
    print 'creating dump folder ' + dump_folder
    os.system('mkdir -p ' + dump_folder)

  # Generate the chord sequence
  parts, chords, num_chords, num_parts = chord_sequence_generation.main(params)
  imgs = two_hot_encoding(parts, chords, num_chords, num_parts)

  blob = {} # output blob which we will dump to JSON for visualizing the results
  blob['params'] = params
  blob['checkpoint_params'] = checkpoint_params
  blob['imgblobs'] = []

  # iterate over all images in test set and predict sentences
  BatchGenerator = decodeGenerator(checkpoint_params)
  n = 0
  candidates=[]
  for img in imgs:
    n+=1
    print 'image %d/%d:' % (n, max_images)
    kwparams = { 'beam_size' : params['beam_size'] }
    img_dict = {'feat': img}
    Ys = BatchGenerator.predict([{'image':img_dict}], model, checkpoint_params, **kwparams)

    # now evaluate and encode the top prediction
    top_predictions = Ys[0] # take predictions for the first (and only) image we passed in
    top_prediction = top_predictions[0] # these are sorted with highest on top
    candidate = ' '.join([ixtoword[ix] for ix in top_prediction[1] if ix > 0]) # ix 0 is the END token, skip that
    candidates.append(candidate)
    print 'PRED: (%f) %s' % (top_prediction[0], candidate)

  # Write midi
  for idx,c in enumerate(candidates):
    cs = c.split()
    for e in cs:
      es=e.split(';')
      pitch=int(es[0])
      pos=es[1]
      pos=convert_pos(pos,idx)
      dur=es[2]
      dur=convert_dur(dur)
      note=pretty_midi.Note(90,pitch,pos,pos+dur)
      new_track.notes.append(note)

  new_midi_data = pretty_midi.PrettyMIDI(initial_tempo=tempo)
  new_midi_data.instruments.append(new_track)

  # pre-set chord preogression
  chord_names = ['C;Em', 'A#;F', 'Dm;Em', 'Dm;G', 'Dm;C', 'Am;Em', 'F;C', 'F;G', 'Dm;F', 'C;C', 'C;E', 'Am;G', 'F;Em', 'F;F', 'G;G', 'Am;Am', 'Dm;Dm', 'C;A#', 'Em;F', 'C;G', 'G#;A#', 'F;Am', 'G#;Fm', 'Am;Gm', 'F;E', 'Dm;Am', 'Em;Em', 'G#;G#', 'Em;Am', 'C;Am', 'F;Dm', 'G#;G', 'F;A#', 'Am;G#', 'C;D', 'G;Am', 'Am;C', 'Am;A#', 'A#;G', 'Am;F', 'A#;Am', 'E;Am', 'Dm;E', 'A;G', 'Am;Dm', 'Em;Dm', 'C;F#m', 'Am;D', 'G#;Em', 'C;Dm', 'C;F', 'G;C', 'A#;A#', 'Am;Caug', 'Fm;G', 'A;A']
  chord_to_pitch = {'C':36, 'C#':37, 'D':38, 'D#':39, 'E':40, 'F':41, 'F#':42, 'G':43, 'G#':44, 'A':45, 'A#':46, 'B':47}
  for time, chord in enumerate(chords):
    n1, n2 = re.split(";", chord_names[chord[0]])
    n1, n2 = re.sub("m", "", n1), re.sub("m", "", n2)
    bass_track.notes.append(pretty_midi.Note(90,chord_to_pitch[n1],2*time,2*time+1))
    bass_track.notes.append(pretty_midi.Note(90,chord_to_pitch[n2],2*time+1,2*(time+1)))
  new_midi_data.instruments.append(bass_track)
  adjust_tempo(new_midi_data)
  if params['quantize']:
    quantize(new_midi_data)
  new_midi_data.write(fout)

def gen_from_test(params):
  # load the checkpoint
  checkpoint_path = params['checkpoint_path']
  max_images = params['max_images']
  fout = params['output_file']
  tempo = params['tempo']

  print 'loading checkpoint %s' % (checkpoint_path, )
  checkpoint = pickle.load(open(checkpoint_path, 'rb'))
  checkpoint_params = checkpoint['params']
  dataset = checkpoint_params['dataset']
  model = checkpoint['model']
  dump_folder = params['dump_folder']

  if dump_folder:
    print 'creating dump folder ' + dump_folder
    os.system('mkdir -p ' + dump_folder)

  # fetch the data provider
  dp = getDataProvider(dataset)

  misc = {}
  misc['wordtoix'] = checkpoint['wordtoix']
  ixtoword = checkpoint['ixtoword']

  blob = {} # output blob which we will dump to JSON for visualizing the results
  blob['params'] = params
  blob['checkpoint_params'] = checkpoint_params
  blob['imgblobs'] = []

  # iterate over all images in test set and predict sentences
  BatchGenerator = decodeGenerator(checkpoint_params)
  n = 0
  all_references = []
  all_candidates = []
  candidates=[]
  for img in dp.iterImages(split = 'test', max_images = max_images):
    n+=1
    print 'image %d/%d:' % (n, max_images)
    references = [' '.join(x['tokens']) for x in img['sentences']] # as list of lists of tokens
    kwparams = { 'beam_size' : params['beam_size'] }

    Ys = BatchGenerator.predict([{'image':img}], model, checkpoint_params, **kwparams)

    img_blob = {} # we will build this up
    img_blob['img_path'] = img['local_file_path']
    img_blob['imgid'] = img['imgid']

    if dump_folder:
      # copy source file to some folder. This makes it easier to distribute results
      # into a webpage, because all images that were predicted on are in a single folder
      source_file = img['local_file_path']
      target_file = os.path.join(dump_folder, os.path.basename(img['local_file_path']))
      os.system('cp %s %s' % (source_file, target_file))

    # encode the human-provided references
    img_blob['references'] = []
    for gtsent in references:
      print 'GT: ' + gtsent
      img_blob['references'].append({'text': gtsent})

    # now evaluate and encode the top prediction
    top_predictions = Ys[0] # take predictions for the first (and only) image we passed in
    top_prediction = top_predictions[0] # these are sorted with highest on top
    candidate = ' '.join([ixtoword[ix] for ix in top_prediction[1] if ix > 0]) # ix 0 is the END token, skip that
    candidates.append(candidate)
    print 'PRED: (%f) %s' % (top_prediction[0], candidate)

    # save for later eval
    all_references.append(references)
    all_candidates.append(candidate)

    img_blob['candidate'] = {'text': candidate, 'logprob': top_prediction[0]}
    blob['imgblobs'].append(img_blob)

  # use perl script to eval BLEU score for fair comparison to other research work
  # first write intermediate files
  print 'writing intermediate files into eval/'
  open('eval/output', 'w').write('\n'.join(all_candidates))
  for q in xrange(1):
    open('eval/reference'+`q`, 'w').write('\n'.join([x[q] for x in all_references]))
  # invoke the perl script to get BLEU scores
  print 'invoking eval/multi-bleu.perl script...'
  owd = os.getcwd()
  os.chdir('eval')
  os.system('./multi-bleu.perl reference < output')
  os.chdir(owd)

  # now also evaluate test split perplexity
  gtppl = eval_split('test', dp, model, checkpoint_params, misc, eval_max_images = max_images)
  print 'perplexity of ground truth words based on dictionary of %d words: %f' % (len(ixtoword), gtppl)
  blob['gtppl'] = gtppl

  # dump result struct to file
  #  print 'saving result struct to %s' % (params['result_struct_filename'], )
  #  json.dump(blob, open(params['result_struct_filename'], 'w'))

  for idx,c in enumerate(candidates):
    cs = c.split()
    for e in cs:
      es=e.split(';')
      pitch=int(es[0])
      pos=es[1]
      pos=convert_pos(pos,idx)
      dur=es[2]
      dur=convert_dur(dur)
      note=pretty_midi.Note(90,pitch,pos,pos+dur)
      new_track.notes.append(note)

  new_midi_data = pretty_midi.PrettyMIDI(initial_tempo=tempo)
  new_midi_data.instruments.append(new_track)

  # pre-set chord preogression
  bass_track.notes.append(pretty_midi.Note(90,36,0,1))
  bass_track.notes.append(pretty_midi.Note(90,47,1,2))
  bass_track.notes.append(pretty_midi.Note(90,45,2,3))
  bass_track.notes.append(pretty_midi.Note(90,43,3,4))
  bass_track.notes.append(pretty_midi.Note(90,41,4,5))
  bass_track.notes.append(pretty_midi.Note(90,40,5,6))
  bass_track.notes.append(pretty_midi.Note(90,38,6,7))
  bass_track.notes.append(pretty_midi.Note(90,43,7,8))

  bass_track.notes.append(pretty_midi.Note(90,36,8,9))
  bass_track.notes.append(pretty_midi.Note(90,47,9,10))
  bass_track.notes.append(pretty_midi.Note(90,45,10,11))
  bass_track.notes.append(pretty_midi.Note(90,43,11,12))
  bass_track.notes.append(pretty_midi.Note(90,41,12,13))
  bass_track.notes.append(pretty_midi.Note(90,40,13,14))
  bass_track.notes.append(pretty_midi.Note(90,38,14,15))
  bass_track.notes.append(pretty_midi.Note(90,43,15,16))

  bass_track.notes.append(pretty_midi.Note(90,45,16,17))
  bass_track.notes.append(pretty_midi.Note(90,41,17,18))
  bass_track.notes.append(pretty_midi.Note(90,36,18,19))
  bass_track.notes.append(pretty_midi.Note(90,43,19,20))
  bass_track.notes.append(pretty_midi.Note(90,45,20,21))
  bass_track.notes.append(pretty_midi.Note(90,41,21,22))
  bass_track.notes.append(pretty_midi.Note(90,43,22,23))
  bass_track.notes.append(pretty_midi.Note(90,43,23,24))

  bass_track.notes.append(pretty_midi.Note(90,36,24,25))
  bass_track.notes.append(pretty_midi.Note(90,47,25,26))
  bass_track.notes.append(pretty_midi.Note(90,45,26,27))
  bass_track.notes.append(pretty_midi.Note(90,43,27,28))
  bass_track.notes.append(pretty_midi.Note(90,41,28,29))
  bass_track.notes.append(pretty_midi.Note(90,40,29,30))
  bass_track.notes.append(pretty_midi.Note(90,38,30,31))
  bass_track.notes.append(pretty_midi.Note(90,43,31,32))

  bass_track.notes.append(pretty_midi.Note(90,36,32,33))
  bass_track.notes.append(pretty_midi.Note(90,47,33,34))
  bass_track.notes.append(pretty_midi.Note(90,45,34,35))
  bass_track.notes.append(pretty_midi.Note(90,43,35,36))
  bass_track.notes.append(pretty_midi.Note(90,41,36,37))
  bass_track.notes.append(pretty_midi.Note(90,40,37,38))
  bass_track.notes.append(pretty_midi.Note(90,38,38,39))
  bass_track.notes.append(pretty_midi.Note(90,43,39,40))

  new_midi_data.instruments.append(bass_track)
  adjust_tempo(new_midi_data)
  if params['quantize']:
    quantize(new_midi_data)
  new_midi_data.write(fout)


def main(params):
  if params["gen_chords"]:
    gen_from_scratch(params)
  else:
    gen_from_test(params)

if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument('checkpoint_path', type=str, help='the input checkpoint')
  parser.add_argument('-b', '--beam_size', type=int, default=1, help='beam size in inference. 1 indicates greedy per-word max procedure. Good value is approx 20 or so, and more = better.')
  parser.add_argument('-m', '--max_images', type=int, default=-1, help='max images to use')
  parser.add_argument('-d', '--dump_folder', type=str, default="", help='dump the relevant images to a separate folder with this name?')
  parser.add_argument('-o', '--output_file', type=str, default="generate_test.mid",help='name of the midi file generated')

  # Chords sequence generation ?
  parser.add_argument('--gen_chords', type=bool, default=False, help='whether the chords and parts are automatically generated or picked from the test set')
  parser.add_argument('--gen_seq_length', type=int, default=8, help='length of the generated sequences')
  parser.add_argument('--nh_part', dest='nh_part', type=int, default=20, help='number of hidden states for the part\'s HMM')
  parser.add_argument('--nh_chords', dest='nh_chords', type=int, default=40, help='number of hidden states for the part\'s HMM')
  parser.add_argument('--num_gen', dest='num_gen', type=int, default=1, help='number of sequences generated')
  parser.add_argument('--quantize', dest='quantize', type=int, default=0, help='')
  parser.add_argument('--tempo', dest='tempo', type=int, default=120, help='beats per minute')

  parser.add_argument('--DEBUG', type=bool, default=False, help='debug mode')

  args = parser.parse_args()
  params = vars(args) # convert to ordinary dict
  print 'parsed parameters:'
  print json.dumps(params, indent = 2)
  main(params)
