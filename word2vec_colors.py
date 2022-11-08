from __future__ import unicode_literals
import spacy

import math
from turtle import dot
import pandas
import random
file_json = pandas.read_json('/Users/daulettoibazar/python projects/word2vec/xkcd.json')


def hex_to_int(s):
    s= s.strip("#")
    return int(s[:2], 16,), int(s[2:4], 16), int(s[4:6], 16)

colors = dict()

for item in file_json["colors"]:
    colors[item["color"]] = hex_to_int(item['hex'])

def distance(coord1, coord2):
    return math.sqrt(sum([(i-j)**2 for i,j in zip(coord1, coord2)]))

def subtractv(coord1, coord2):
    return [c1-c2 for c1,c2 in zip(coord1, coord2)]


def addv(coord1, coord2):
    return [c1+c2 for c1,c2 in zip(coord1, coord2)]


def meanv(coords):
    sumv = [0] * len(coords[0])
    for item in coords:
        for i in range(len(item)):
            sumv[i] +=item[i]
    mean = [0] * len(sumv)
    for i in range(len(sumv)):
        mean[i] = float(sumv[i])/len(coords)
    return mean

def closest(space, coord, n =3):
    colsest = []
    for key in sorted(space.keys(), key = lambda x:distance(coord, space[x]))[:6]:
        colsest.append(key)
    return colsest


red = colors["red"]
blue = colors["blue"]
for i in range(3):
    rednames = closest(colors,red)
    bluenames = closest(colors,blue)
    print("Roses are",rednames[0],"violets are", bluenames[0])
    red = colors[random.choice(rednames[1:])]
    blue = colors[random.choice(bluenames[1:])]

path = '/Users/daulettoibazar/python projects/word2vec/pg345.txt'

file_text = open(path)
content = ""
for lines in file_text:
    content+=lines
file_text.close()

from spacy import vocab
nlp = spacy.load('en_core_web_lg')

doc = nlp(content)
tokens = list(set([w.text for w in doc if w.is_alpha]))

def vec(s):
       return nlp(s).vector
import numpy
from numpy import dot
from numpy.linalg import norm

def cosine(v1, v2):
       if norm(v1)>0 and norm(v2)>0:
              return dot(v1,v2)/(norm(v1)*norm(v2))
       else:
              return 0.0
print(cosine(vec('dog'),vec('cat'))>cosine(vec('trousers'),vec('skirt')))

def spacy_closest(token_list,vec_to_check, n = 10):
       return sorted(token_list, key = lambda x: cosine(vec_to_check, vec(x)), reverse = True)[:n]

print(spacy_closest(tokens, vec('basketball')))
print(spacy_closest(tokens, vec("grass")))
print(spacy_closest(tokens, subtractv(vec("wine"), vec("alcohol"))))

def sentvec(s):
       sent = nlp(s)
       return meanv([w.vector for w in sent])

sentences = list(doc.sents)
