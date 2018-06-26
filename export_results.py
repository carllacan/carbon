# -*- coding: utf-8 -*-
#import jinja2
#import os
#from jinja2 import Template

#runs = (4,16,27,36,51,63,75,87,99) 
#runs = (10,22,35,46,57,70,82,94,99) 

runlist = (4, 15, 24, 36, 51, 62, 77, 84, 96)
#runlist = (9, 22, 34, 43, 55, 71, 81, 93, 96)

# TODO: make this script eport also the times table

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
        res = values[2:]
        results.append(res)
         
t = open('latex/templates/tabletemp.tex')
dest = open('latex/' + '/result_1repeat.tex', 'w')

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