# -*- coding: utf-8 -*-
import jinja2
import os
from jinja2 import Template

runs = (0, 0, 0, 1, 1, 1, 2, 3, 4)
delimiter = ','
colignore = 11
enum = 3 # number of errors

fname = 'results_fixed_seed_bestmodels.csv'
f = open(fname, 'r')

results = []
#for l in f.readlines()[1:]:
entries = f.readlines()[1:]
for r in runs:#f.readlines()[1:]:
    l = entries[r]
    errors = l.strip('\n').split(delimiter)[colignore:-1]
    # append each group of results
    for i in range(0, len(errors), enum):
        results.append(errors[i:i+enum])
        results[-1].append('')
     
#latex_jinja_env = jinja2.Environment(
#	block_start_string = '\BLOCK{',
#	block_end_string = '}',
#	variable_start_string = '\VAR{',
#	variable_end_string = '}',
#	comment_start_string = '\#{',
#	comment_end_string = '}',
#	line_statement_prefix = '%%',
#	line_comment_prefix = '%#',
#	trim_blocks = True,
#	autoescape = False,
#	loader = jinja2.FileSystemLoader(os.path.abspath('.'))
#)
#template = latex_jinja_env.get_template('tabletemp.tex')
#
#print(template.render(results=results))
#        
#t = open('tabletemp2.tex')
##skiplines = 6
#dest = open('result_summary.tex', 'w')
#string_template = '& {} '*4
#for i, l in enumerate(t.readlines()):
#    line = l.strip('\n')
#    if 5 < i and i < 24:
#        errors = results[i-6] + ['']
##        line += '& {} & {} & {} & {}'.format(*errors)
#        line += string_template.format(*errors)
#        line += '\\\\ \cline{ 3- 6}'
#    dest.write(line + '\n')
#    
#dest.close()
#t.close()
    
    
t = open('tabletemp.tex')
dest = open('result_summary.tex', 'w')

i = 0
for l in t.readlines():
    line = l
    if ' {} ' in l:
        for j in range(l.count(' {} ')):
            error = results[i][j]
            stringed = ' {} '.format(error)
            line = line.replace(' {} ', stringed)
        i += 1
    dest.write(line + '\n')
    
dest.close()
t.close()
    
    
    
    
    
    
    
    
    
    
