import os
import re
import random
from copy import deepcopy

from sklearn.utils import shuffle

def get_scale(tonic, notes):
    n = 0
    for i in range(len(notes)):
        if notes[i] == tonic[0]:
            n = i
            break
    notes = notes[n:] + notes[:n]
    notes_dict = {}
    for i in range(1,8):
        notes_dict[i] = notes[i - 1]
    return notes_dict

def compute_duration(note, duration):
    for char in list(note):
        if char == '.':
            duration *= 1.5
        elif char == '_':
            duration *= 2
        else:
            duration *= 1
    return duration

def compute_pitch(pitch, octave, scale, duration, octave_level):
    p = pitch
    if pitch == '0':
        return str(duration)
    else:
        for char in list(p):
            if char == '-':
                octave /= 2
                pitch = pitch.replace(char, '')
            elif char == '+':
                octave *= 2
                pitch = pitch.replace(char, '')
            elif char == 'b' or char == '#':
                octave *= 1
            else:
                pitch = pitch.replace(char, scale[int(char)])
        freq = octave_level[pitch] * octave
        pitch = str(freq) + str(duration)
        return pitch

def preprocess_music():    
    path = './music_dataset'
    files = os.listdir(path)
    music_notes = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    #Frequencies in the sub-contra octave
    sub_contra = {'A' : 27.5, 'A#' : 29.14, 'Bb' : 29.14, 'B' : 30.87, 'B#' : 16.35, 'C' : 16.35, 'C#' : 17.32, 'Cb' : 30.87,
                  'Db' : 17.32, 'D' : 18.35, 'D#' : 19.45, 'Eb' : 19.45, 'E' : 20.60, 'E#' : 21.38, 'F' : 21.83, 'Fb' : 20.6,
                  'F#' : 23.12, 'Gb' : 23.13, 'G' : 24.5, 'G#' : 25.96, 'Ab' : 25.96}

    mels = []
    mels_shuflled = []
    labels = []
    for file in files:
        f = open(path+'/'+file, 'r', encoding = 'cp437')
        music = f.read()
        r1 = r'(?<=MEL\[)[^/\]]+'
        r2 = r'(?<=KEY\[)[^\]]+(?=\])'
        melodies = re.findall(r1, music, re.DOTALL)
        keys = re.findall(r2, music, re.DOTALL)
        for i in range(len(keys)):

            #initialize
            mel = []
            label = [1]
            r3 = r'[+-]*[0-7][b#]*[_.]*|\n'
            notes = re.findall(r3, melodies[i], re.DOTALL)
            k = keys[i].replace('.',' ').split()
            shortest_rhytmic_unit = int(k[1])
            tonic = k[2]
            scale = get_scale(tonic, music_notes)

            #calculation for the first note
            absolute_duration = compute_duration(notes[0], shortest_rhytmic_unit)
            pitch = re.findall(r'([+-]*[0-7][b#]*)[_.]*', notes[0])
            if pitch:
                pitch = pitch[0]
                pitch = compute_pitch(pitch, 1, scale, absolute_duration, sub_contra)
                mel.append(pitch)

            #calculations for the rest of the notes
            for j in range(1, len(notes)):
                prev = notes[j - 1]
                note = notes[j]
                if prev == '\n':
                    lbl = 1
                else:
                    lbl = 0
                if note == '\n':
                    continue
                else:
                    label.append(lbl)
                    r4 = r'([+-]*[0-7][b#]*)[_.]*'
                    pitch = re.findall(r4, note)
                    absolute_duration = compute_duration(note, shortest_rhytmic_unit)
                    if pitch:
                        pitch = pitch[0]
                        pitch = compute_pitch(pitch, 1, scale, absolute_duration, sub_contra)
                        mel.append(pitch)
            
            mels.append(mel)
            shuffle_mel = deepcopy(mel)
            random.shuffle(shuffle_mel)
            mels_shuflled.append(shuffle_mel)
            labels.append(label)
    return mels, mels_shuflled, labels

X_music, X_shuffled_music, Y_music = preprocess_music()

X_unique = []
for i in range(len(X_music)):
    for item in X_music[i]:
        if (item in X_unique) == False:
            X_unique.append(item)

number_list = []
for i in range(len(X_unique)):
    number_list.append(i)
X_unique.sort()
note2id = dict(zip(X_unique, number_list))