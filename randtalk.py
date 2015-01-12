# randtalk.py
# implementation of nettalk using random indexing on nettalk dataset

# libraries
import random_idx
import utils
import sys
import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt
import string
from tqdm import tqdm

# fixed parameters
N = 10000
k = 10
window_sz = 2
alphabet = string.lowercase + ' .'
phonemes = 'abcdefghiklmnoprstuvwxyz' + 'ACDEGIJKLMNORSTUWXYZ' + '@!#*^+-'
stresses = '><012'
datafile = './data/nettalk_m.data'
num_epochs = 1

# initialize RI vectors
RI_letters = random_idx.generate_letter_id_vectors(N, k, alph=alphabet)
phonemic_vecs = np.zeros((len(phonemes),N))
stress_vecs = np.zeros((len(stresses),N))

# debugging parameters
i = 0
debug_num = 0

def buffed(in_string, window_sz=window_sz):
    # pad string with ~window_sz~ spaces on front and back
    buffed_string = in_string
    for b in xrange(window_sz):
            buffed_string = ' ' + buffed_string + ' '
    return buffed_string

def update_PS_vec(word, phoneme, stress, display=0):
    # update global phoneme and stress vectors
    if display:
            print word
    # add ~window_sz~ space buffer to word, phoneme, and stress
    buffed_word = buffed(word)
    buffed_phoneme = buffed(phoneme)
    buffed_stress = buffed(stress)

    # iterate through letters in ~buffed_word~ (not ~word~ to avoid string indexing error!)
    for letter_idx in xrange(window_sz, len(buffed_word)-window_sz):
            window = buffed_word[letter_idx - window_sz: letter_idx + window_sz + 1]
            if display:
                    print window, buffed_word[letter_idx], buffed_phoneme[letter_idx], buffed_stress[letter_idx]
            phonem_vec, stress_vec = create_PS_win(window)
            phonem = phonemes.index(buffed_phoneme[letter_idx])
            stres = stresses.index(buffed_stress[letter_idx])
            phonemic_vecs[phonem,:] = np.reshape(phonemic_vecs[phonem,:],(1,N)) + phonem_vec
            stress_vecs[stres,:] = np.reshape(stress_vecs[stres,:],(1,N)) + stress_vec

def create_PS_win(window,window_sz=window_sz):
    # create a phoneme and stress vector given a window of window_sz
    phonem_vec = np.zeros((1,N))
    stress_vec = np.zeros((1,N))
    roller = 0
    for letter in window:
            alph_idx = alphabet.index(letter)
            alph_vec = np.roll(RI_letters[alph_idx,:], -window_sz + roller)
            phonem_vec += alph_vec
            stress_vec += alph_vec
            roller += 1
    return phonem_vec, stress_vec

def test_PS_vec(word, phoneme, stress, display=0):
    if display:
            print word
    buffed_word = buffed(word)
    buffed_phoneme = buffed(phoneme)
    buffed_stress = buffed(stress)

    for letter_idx in xrange(window_sz, len(buffed_word)-window_sz):
            window = buffed_word[letter_idx - window_sz: letter_idx + window_sz + 1]
            if display:
                    print "~~~~~~~~~~~~"
                    print window, buffed_word[letter_idx], buffed_phoneme[letter_idx], buffed_stress[letter_idx]
            phonem_vec, stress_vec = create_PS_win(window)
            #phonem_vec_n /= np.linalg.norm(phonem_vec)
            #stress_vec_n /= np.linalg.norm(stress_vec)
            phonem = phonemes.index(buffed_phoneme[letter_idx])
            stres = stresses.index(buffed_stress[letter_idx])
            likely_phonem, phonem_angs = utils.find_language(buffed_phoneme[letter_idx], phonem_vec, phonemic_vecs, list(phonemes),display=1)
            likely_stres, stress_angs = utils.find_language(buffed_stress[letter_idx], stress_vec, stress_vecs, list(stresses),display=1)

            # add number of correct and total phonemes/stress for results
            global total_phoneme; global total_stress; global correct_phoneme; global correct_stress
            total_phoneme += 1
            total_stress += 1
            if buffed_phoneme[letter_idx] == likely_phonem:
                    correct_phoneme += 1
            if buffed_stress[letter_idx] == likely_stres:
                    correct_stress += 1

# train phonemic and stress syllabic vectors
for i in xrange(num_epochs):
        data = open(datafile)
        for line in tqdm(data, total=20007):
            word, phoneme, stress, odd = string.split(line,'\t')
            update_PS_vec(word,phoneme,stress)

            # debugging line
            #if i >= debug_num: break

# normalize learnt vectors
#for j in xrange(len(phonemes)):
#    phonemic_vecs[j,:] /= np.linalg.norm(phonemic_vecs[j,:])
#for j in xrange(len(stresses)):
#    stress_vecs[j,:] /= np.linalg.norm(stress_vecs[j,:])

phonangles = utils.cosangles(phonemic_vecs,list(phonemes))
#print phonangles

stresangles = utils.cosangles(stress_vecs,list(stresses))
#print stresangles

# test
testing = 0
test_num = 1
total_phoneme = 0
total_stress = 0
correct_phoneme = 0
correct_stress = 0
data = open(datafile)
for line in data:
    word, phoneme, stress, odd = string.split(line,'\t')
    test_PS_vec(word, phoneme, stress, display=1)
    if testing >= test_num: break
    testing += 1

print 'phoneme correctness: %4.4f' % (float(correct_phoneme)/total_phoneme)
print 'stress correctness: %4.4f' % (float(correct_stress)/total_stress)
