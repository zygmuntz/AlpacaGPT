#!/usr/bin/env python

"convert alpaca data from instruction-input-output format to instruction-output format"

import json
from munch import munchify

input_file = 'data/alpaca_data_cleaned.json'
output_file = 'data/modal/train.json'

with open( input_file, 'r', encoding = 'utf-8' ) as i_f:
	examples = munchify( json.load( i_f ))
	
converted = []	
for example in examples:

	glue_chars = ''
	if example.input != '':
		end_char = example.instruction[-1] 
		
		if end_char not in ( '.', '?', ':' ):
			if example.input[0].isupper():
				glue_chars = '.'
			else:
				glue_chars = ':'
				
		new_example = { 'instruction': '{}{} {}'.format( example.instruction, glue_chars, example.input ), 'output': example.output }
	else:
		new_example = { 'instruction': example.instruction, 'output': example.output }
	
	converted.append( new_example )

with open( output_file, 'w', encoding = 'utf-8' ) as o_f:
	json.dump( converted, o_f, indent = 4 )
	
