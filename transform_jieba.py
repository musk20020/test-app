# -*- coding: utf8 -*-

'''
iob format generateor
'''

import xlrd,xlwt
from collections import OrderedDict
import simplejson as json
import urllib2
import time
import jieba

# jieba dict
jieba_dict = "./jieba-zh_TW/dict.txt"
corpus_dict = "./jieba-zh_TW/corpus_jieba.txt"
jieba.set_dictionary(jieba_dict)
jieba.load_userdict(corpus_dict)
#open test file  

# open_parameter = xlrd.open_workbook('./utterance/ijia_utterance.xls')
# sheet_index = 3

open_parameter = xlrd.open_workbook('./utterance/primary_utterance_UT_0830.xlsx')
sheet_index = 0

output_workbook = xlwt.Workbook()
wt_sheet = output_workbook.add_sheet('jieba_sheet')
wt_style = xlwt.XFStyle()
wt_style.alignment.wrap = 1

tStart = time.time()

# write header
wt_sheet.write(0, 0, "ID")
wt_sheet.write(0, 1, "Utterance")
wt_sheet.write(0, 2, "Intent")
wt_sheet.write(0, 3, "Entities")

line_count = 0

while True:
    try:
        parameter_sheet = open_parameter.sheet_by_index(sheet_index)   #read sheet
        for rownum in range(1, parameter_sheet.nrows):
            row_values = parameter_sheet.row_values(rownum)
            if (len(row_values[1]) > 0 ):
                _term = row_values[1]

                try:
                    print rownum, _term
                except:
                    print rownum, "item can't print out"
                    
                _t = _term.encode('UTF-8')

                # jieba parsed
                jiebad_utter = jieba.cut(_t)
                jiebad_utter = (u' '.join(jiebad_utter))

                line_count += 1
                wt_sheet.write(line_count, 1, 'BOS ' + jiebad_utter + ' EOS')   # sentence
                wt_sheet.write(line_count, 2, row_values[2])   # intent
                wt_sheet.write(line_count, 3, row_values[3])   # entity
                
                output_workbook.save('./utterance/primary_seg_out.xls')
        sheet_index += 1
    except Exception:
        print 'exception'
        break


tEnd = time.time()
