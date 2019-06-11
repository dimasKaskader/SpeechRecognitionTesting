import os

from deepspeech_recognizer import DeepSpeech
from kaldiasr.nnet3 import KaldiNNet3OnlineModel, KaldiNNet3OnlineDecoder
from openseq2seq_recognizer import OpenSeq2Seq
from wer import wer


DEEPSPEECH_MODEL = './deepspeech/models'
KALDI_MODEL = './kaldi/models/kaldi-generic-en-tdnn_sp-r20180815'
OPENSEQ2SEQ_MODEL = './OpenSeq2Seq/Infer_S2T W2L'

deepspeech = DeepSpeech(DEEPSPEECH_MODEL)
kaldi_model = KaldiNNet3OnlineModel(KALDI_MODEL, acoustic_scale=1.0, beam=7.0, frame_subsampling_factor=3)
kaldi_decoder = KaldiNNet3OnlineDecoder(kaldi_model)
openseq2seq = OpenSeq2Seq(OPENSEQ2SEQ_MODEL)


def kaldi_recognize(wav_file):
    if kaldi_decoder.decode_wav_file(wav_file):
        s, l = kaldi_decoder.get_decoded_string()
        return s
    else:
        return "***ERROR: decoding of %s failed." % wav_file


def append_to_file(file, line):
    with open(file, 'a') as f:
        f.write(line + '\n')


AUDIO_DIR = 'audio'
with open(AUDIO_DIR + '/' + 'files.csv', 'r') as files:
    for file in files.readlines():
        splitter = file.split(',')
        if splitter[1][-1] == '\n':
            splitter[1] = splitter[1][0:-1]
        audio_file = AUDIO_DIR + '/' + splitter[0]
        recognized_text = deepspeech.recognize(audio_file)
        print(recognized_text)
        w, r = wer(splitter[1].split(), recognized_text.split())
        append_to_file('deepspeech.csv', recognized_text + ',' + splitter[1] + ',' + w + ',' + r)
        '''recognized_text = kaldi_recognize(audio_file)
        w, r = wer(splitter[1].split(), recognized_text.split())
        append_to_file('kaldi.csv', recognized_text + ',' + splitter[1] + ',' + w + ',' + r)
        recognized_text = openseq2seq.recognize(audio_file)
        w, r = wer(splitter[1].split(), recognized_text.split())
        append_to_file('openseq2seq.csv', recognized_text + ',' + splitter[1] + ',' + w + ',' + r)'''




