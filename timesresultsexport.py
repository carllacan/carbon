#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 17:11:24 2018

@author: carles
"""

# -*- coding: utf-8 -*-
#import jinja2
#import os
#from jinja2 import Template

#runs = (4,16,27,36,51,63,75,87,99) 
#runs = (10,22,35,46,57,70,82,94,99) 

runlist = (4, 15, 24, 36, 51, 62, 77, 84, 96)
#runlist = (9, 22, 34, 43, 55, 71, 81, 93, 96)

resultsfolder = 'results/results1_repeat'
results_filename = resultsfolder + '/results.csv'
delimiter = ','
colignore = 11
enum = 3 # number of errors

f = open(results_filename, 'r')

results = []
lines = f.readlines()[1:]
for l in lines:
    values = l.strip('\n').split(delimiter)
    if int(values[0]) in runlist:
        res = values[-2:]
        results.append(res)
         
t = open('latex/templates/timestabletemp.tex')
dest = open('latex/' + '/result_1repeat_times.tex', 'w')

i = 0
for l in t.readlines():
    line = l
    if 'VAR' in l:
        for j in range(l.count('VAR')):
            error = results[i][j]
            stringed = '{:2.3f}'.format(float(error))
            line = line.replace('VAR', stringed)
        i += 1
#    print(line)
    dest.write(line + '\n')
    
dest.close()
t.close()