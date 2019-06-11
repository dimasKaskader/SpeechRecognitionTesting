import speech_recognition as sr
import urllib.request as req
from xml.dom import minidom
from os import path
import os
import deep_speech as ds


def recognize_yandex(audio):
    key = 'abc41255-8098-4fb0-8f6f-45be137bfc05'
    uuid = 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaab'

    data = audio.get_wav_data()
    request = req.Request(url='https://asr.yandex.net/asr_xml?uuid=' + uuid + '&key=' + key + '&topic=queries&lang=en-US',
                          headers={'Content-Type': 'audio/x-wav', 'Content-Length': len(data)})
    response = req.urlopen(url=request, data=data)

    xmldoc = minidom.parseString(response.read())
    if xmldoc.getElementsByTagName('recognitionResults')[0].attributes['success'].value == '1':
        '''for variant in xmldoc.getElementsByTagName('variant'):
            print(variant.attributes['confidence'].value + ' ' + variant.childNodes[0].nodeValue)'''
        return xmldoc.getElementsByTagName('variant')[0].childNodes[0].nodeValue


AUDIO_DIR = path.join(path.dirname(path.realpath(__file__)), 'audio')
files = os.listdir(AUDIO_DIR)
for file in files:
    AUDIO_FILE = AUDIO_DIR + '/' + file
    print()
    print('File: ' + file)
    # use the audio file as the audio source
    r = sr.Recognizer()
    with sr.AudioFile(AUDIO_FILE) as source:
        audio = r.record(source)  # read the entire audio file

    # recognize speech using Sphinx
    try:
        print("PocketSphinx: " + r.recognize_sphinx(audio))
    except sr.UnknownValueError:
        print("Sphinx could not understand audio")
    except sr.RequestError as e:
        print("Sphinx error; {0}".format(e))

    # recognize speech using Google Speech Recognition
    try:
        # for testing purposes, we're just using the default API key
        # to use another API key, use `r.recognize_google(audio, key="GOOGLE_SPEECH_RECOGNITION_API_KEY")`
        # instead of `r.recognize_google(audio)`
        print("Google Speech Recognition: " + r.recognize_google(audio))
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))

    print("Yandex SpeechKit: " + recognize_yandex(audio))
    print("Mozilla DeepSpeech: " + ds.recognize_deepspeech(AUDIO_FILE))

