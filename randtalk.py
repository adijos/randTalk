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

# fixed parameters
N = 1000
k = 5
window_sz = 3
alphabet = string.lowercase + ' .'
phonemes = 'abcdefghiklmnoprstuvwxyz' + 'ACDEGIJKLMNORSTUWXYZ' + '@!#*^+-'
stresses = '><012'
datafile = './data/nettalk_m.data'
num_epochs = 3

# initialize RI vectors
RI_letters = random_idx.generate_letter_id_vectors(N, k, alph=alphabet)
phonemic_vecs = np.zeros((len(phonemes),N))
stress_vecs = np.zeros((len(stresses),N))

# debugging parameters
i = 0
debug_num = 0

def update_PS_vec(word, phoneme, stress):
    print word
    buffed_word = '   ' + word + '   '
    buffed_phoneme = '   ' + phoneme + '   '
    buffed_stress = '   ' + stress + '   '
    for letter_idx in xrange(window_sz, len(buffed_word)-window_sz):
            window = buffed_word[letter_idx - window_sz: letter_idx + window_sz + 1]
            print window, buffed_word[letter_idx], buffed_phoneme[letter_idx], buffed_stress[letter_idx]
            phonem_vec, stress_vec = create_PS_win(window)
            phonem = phonemes.index(buffed_phoneme[letter_idx])
            stres = stresses.index(buffed_stress[letter_idx])
            phonemic_vecs[phonem,:] = np.reshape(phonemic_vecs[phonem,:],(1,N)) + phonem_vec
            stress_vecs[stres,:] = np.reshape(stress_vecs[stres,:],(1,N)) + stress_vec

def create_PS_win(window,window_sz=window_sz):
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

def test_PS_vec(word, phoneme, stress):
    print word
    buffed_word = '   ' + word + '   '
    buffed_phoneme = '   ' + phoneme + '   '
    buffed_stress = '   ' + stress + '   '
    for letter_idx in xrange(window_sz, len(buffed_word)-window_sz):
            window = buffed_word[letter_idx - window_sz: letter_idx + window_sz + 1]
            print window, buffed_word[letter_idx], buffed_phoneme[letter_idx], buffed_stress[letter_idx]
            phonem_vec, stress_vec = create_PS_win(window)
            #phonem_vec_n /= np.linalg.norm(phonem_vec)
            #stress_vec_n /= np.linalg.norm(stress_vec)
            phonem = phonemes.index(buffed_phoneme[letter_idx])
            stres = stresses.index(buffed_stress[letter_idx])
            likely_phonem = utils.find_language(buffed_phoneme[letter_idx], phonem_vec, phonemic_vecs, list(phonemes),display=1)
            likely_stres = utils.find_language(buffed_stress[letter_idx], stress_vec, stress_vecs, list(stresses),display=1)

# train phonemic and stress syllabic vectors
for i in xrange(num_epochs):
        data = open(datafile)
        for line in data:
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
print phonangles

stresangles = utils.cosangles(stress_vecs,list(stresses))
print stresangles

# test
testing = 0
test_num = 0
data = open(datafile)
for line in data:
    word, phoneme, stress, odd = string.split(line,'\t')
    test_PS_vec(word, phoneme, stress)
    if testing >= test_num: break
    testing += 1

