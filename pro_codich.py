import speech_recognition as sr
import urllib.request as req
from xml.dom import minidom
from os import path
import os
import openpyxl


'''class Excel:
    index = 2
    wb = openpyxl.load_workbook(filename='excel.xlsx')
    sheet = wb['table1']

    @staticmethod
    def init():
        while Excel.sheet['A' + str(Excel.index)].value is not None:
            Excel.index += 1

    @staticmethod
    def write_line(name, sphinx, yandex, google):

        sheet = Excel.sheet

        sheet['A' + str(Excel.index)] = name
        sheet['B' + str(Excel.index)] = sphinx
        sheet['E' + str(Excel.index)] = '-'
        sheet['H' + str(Excel.index)] = yandex
        sheet['K' + str(Excel.index)] = google
        Excel.index += 1

    @staticmethod
    def close():
        Excel.wb.save('excel.xlsx')'''


def recognize_yandex(audio):
    key = '1e692527-ad23-4fdb-b463-b34e545f9a13'
    uuid = 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaab'

    data = audio.get_wav_data()
    request = req.Request(url='https://asr.yandex.net/asr_xml?uuid=' + uuid + '&key=' + key + '&topic=queries',
                          headers={'Content-Type': 'audio/x-wav', 'Content-Length': len(data)})
    response = req.urlopen(url=request, data=data)

    xmldoc = minidom.parseString(response.read())
    if xmldoc.getElementsByTagName('recognitionResults')[0].attributes['success'].value == '1':
        return xmldoc.getElementsByTagName('variant')[0].childNodes[0].nodeValue


AUDIO_DIR = path.join(path.dirname(path.realpath(__file__)), 'audio')
r = sr.Recognizer()
files = os.listdir(AUDIO_DIR)
for file in files:
    AUDIO_FILE = AUDIO_DIR + '/' + file
    print()
    print('File: ' + file)
    with sr.AudioFile(AUDIO_FILE) as source:
        audio = r.record(source)   # чтение аудиофайла

    sphinx = r.recognize_sphinx(audio)    #распознавание с помощью sphinx
    print("PocketSphinx: " + sphinx)

    google = r.recognize_google(audio, language='ru')   #распознавание с помощью google
    print("Google Speech Recognition: " + google)

    yandex = recognize_yandex(audio)    #распознавание с помощью яндекс
    print("Yandex SpeechKit: " + yandex)

    #Excel.write_line(file.split('.')[0], sphinx, yandex, google)

#Excel.close()

