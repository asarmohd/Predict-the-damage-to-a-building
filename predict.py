# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 18:24:23 2018

@author: ma
"""

import pandas as pd
import sklearn as sk
import matplotlib.pyplot as plt

data = pd.read_csv("train.csv")
y = data.iloc[:,2]
X = data.iloc[:,:]
sample_X = X.sample(n = 500000) 
grade = []

for val in sample_X['damage_grade']:
    cool = int(str(val).replace("Grade ",''))
    grade.append(cool)

del(sample_X['damage_grade'])

    
dict_area = {}    
count = 0
for i in set(sample_X['area_assesed']):
    dict_area[i] =  count
    count = count+1
print(dict_area)   

arear_details = []

for val in sample_X['area_assesed']:
    cool = dict_area[val]
    arear_details.append(cool)
    
sample_X['area_assesed'] = arear_details

build_id = sample_X['building_id'] 
del(sample_X['building_id'])
del(sample_X['has_repair_started'])

#process grade
    

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 21, metric = 'minkowski', p = 2)
classifier.fit(sample_X, grade)

#predict-------


import pandas as pd
import sklearn as sk
import matplotlib.pyplot as plt

data = pd.read_csv("test.csv")
X = data.iloc[:,:]
sample_X = X
    
dict_area = {}    
count = 0
for i in set(X['area_assesed']):
    dict_area[i] =  count
    count = count+1
print(dict_area)   

arear_details = []

for val in  X['area_assesed']:
    cool = dict_area[val]
    arear_details.append(cool)
    
X['area_assesed'] = arear_details

build_id = sample_X['building_id'] 
del(sample_X['building_id'])
del(sample_X['has_repair_started'])



y_pred = classifier.predict(sample_X)

df = pd.DataFrame()
grad_p = []
for i in y_pred:
    grai = 'Grade '+str(i)
    grad_p.append(grai)
df['building_id'] = build_id
df['damage_grade'] = grad_p
df.to_csv('out1.csv')
