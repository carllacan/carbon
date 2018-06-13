#!/usr/bin/env python3
# -*- coding: utf-8 -*-

target_folder = 'results/results1'
origin = 'results.csv'
dest = 'results_denorm.csv'

origin_file = open(target_folder + '/' + origin)
dest_file = open(target_folder + '/' + dest, 'w')

stds = (2.895,
        1.654,
        1.911,
        2.413,
        5.259,
        3.243)
delimiter = ','
for l in origin_file.readlines()[1:]:
    vs = l.split(delimiter)
    target = int(vs[1])
    dnorm = lambda x: '{:f}'.format(float(x)*std)
    std = stds[target]
    w = delimiter.join((vs[0], 
                       vs[1],
                       dnorm(vs[2]),
                       dnorm(vs[3]),
                       dnorm(vs[4]),
                       vs[5],
                       vs[6],
                       vs[7]))
#    print(l)
#    print(w)
    dest_file.write(w + '\n')
    
    

