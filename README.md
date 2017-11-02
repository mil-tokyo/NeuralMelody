## Overview

- This repository implements melody generation model proposed in [this paper](https://arxiv.org/abs/1710.11549).

- The **input** is a two-hot vector in which the first 1 corresponds to a certain chord progression of 2-bar lengths (ex: C - Am), and the second 1 corresponds to the part annotation, e.g., verse, chorus, etc.

- The **output** is a MIDI file with generated melody converted from generated strings. Generated strings are currently in the form of **pitch;pos;duration**.

-  This repository is a modification of [NeuralTalk](https://github.com/karpathy/neuraltalk).


## Dependencies
- **[pretty-midi](https://github.com/craffel/pretty-midi)**

- **[hmmlearn](https://github.com/hmmlearn/hmmlearn)**

- **[mido](http://mido.readthedocs.io/en/latest/installing.html)** 

## Usage
- To train

  `python train.py`

- To deactivate regularization on pitch range

  `python train.py --reg_range_coeff 0`

- To set pitch range for regularization (default is 60~72)

  `python train.py --reg_range_min your_min_val --reg_range_max your_max_val`

- To generate MIDI file

  `python generate_midi.py cv/checkpoint_file`

- To generate MIDI file with HMM-generated input (by default, song will be generated based on our pre-set test input)

  `python generate_midi.py cv/checkpoint_file --gen_chords True`


- Notes are inserted to MIDI files on a real-valued time instead of discrete musical lengths, so make sure to quantize it on any sequencer (e.g. GarageBand). 1/16 is recommended. 

- Check our [demos](https://soundcloud.com/iclr2018eval)

## Citation
`@article{andrew2017neuralmelody,
    author={Andrew Shin, Leopold Crestel, Hiroharu Kato, Kuniaki Saito, Katsunori Ohnishi, Masataka Yamaguchi, Masahiro Nakawaki, Yoshitaka Ushiku, Tatsuya Harada},
    title={Melody Generation for Pop Music via Word Representation of Musical Properties},
    journal={arXiv preprint arXiv:1710.11549},
    year={2017}
}`

## License
BSD license.
