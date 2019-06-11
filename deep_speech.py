#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

from os import path
from timeit import default_timer as timer

import argparse
import subprocess
import sys
import scipy.io.wavfile as wav
import numpy as np
from deepspeech.model import Model

# These constants control the beam search decoder

# Beam width used in the CTC decoder when building candidate transcriptions
BEAM_WIDTH = 500

# The alpha hyperparameter of the CTC decoder. Language Model weight
LM_WEIGHT = 1.75

# The beta hyperparameter of the CTC decoder. Word insertion weight (penalty)
WORD_COUNT_WEIGHT = 1.00

# Valid word insertion weight. This is used to lessen the word insertion penalty
# when the inserted word is part of the vocabulary
VALID_WORD_COUNT_WEIGHT = 1.00


# These constants are tied to the shape of the graph used (changing them changes
# the geometry of the first layer), so make sure you use the same constants that
# were used during training

# Number of MFCC features to use
N_FEATURES = 26

# Size of the context window used for producing timesteps in the input vector
N_CONTEXT = 9


def convert_samplerate(audio_path):
    sox_cmd = 'sox {} --type raw --bits 16 --channels 1 --rate 16000 - '.format(audio_path)
    try:
        p = subprocess.Popen(sox_cmd.split(),
                             stderr=subprocess.PIPE, stdout=subprocess.PIPE)
        output, err = p.communicate()

        if p.returncode:
            raise RuntimeError('SoX returned non-zero status: {}'.format(err))

    except OSError as e:
        raise OSError('SoX not found, use 16kHz files or install it: ', e)

    audio = np.fromstring(output, dtype=np.int16)
    return 16000, audio


def recognize_deepspeech(audio):

    model = path.join(path.dirname(path.realpath(__file__)), 'models/output_graph.pb')
    alphabet = path.join(path.dirname(path.realpath(__file__)), 'models/alphabet.txt')
    lm = path.join(path.dirname(path.realpath(__file__)), 'models/lm.binary')
    trie = path.join(path.dirname(path.realpath(__file__)), 'models/trie')

    #print('Loading model from file %s' % (model), file=sys.stderr)
    model_load_start = timer()
    ds = Model(model, N_FEATURES, N_CONTEXT, alphabet, BEAM_WIDTH)
    model_load_end = timer() - model_load_start
    #print('Loaded model in %0.3fs.' % (model_load_end), file=sys.stderr)

    if lm and trie:
        #print('Loading language model from files %s %s' % (lm, trie), file=sys.stderr)
        lm_load_start = timer()
        ds.enableDecoderWithLM(alphabet, lm, trie, LM_WEIGHT,
                               WORD_COUNT_WEIGHT, VALID_WORD_COUNT_WEIGHT)
        lm_load_end = timer() - lm_load_start
        #print('Loaded language model in %0.3fs.' % (lm_load_end), file=sys.stderr)

    fs, audio = wav.read(audio)
    if fs != 16000:
        if fs < 16000:
            print('Warning: original sample rate (%d) is lower than 16kHz. Up-sampling might produce erratic speech recognition.' % (fs), file=sys.stderr)
        fs, audio = convert_samplerate(audio)
    audio_length = len(audio) * ( 1 / 16000)

    #print('Running inference.', file=sys.stderr)
    #inference_start = timer()
    #inference_end = timer() - inference_start
    return ds.stt(audio, fs)
    #print('Inference took %0.3fs for %0.3fs audio file.' % (inference_end, audio_length), file=sys.stderr)

