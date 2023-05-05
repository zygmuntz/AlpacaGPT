#!/usr/bin/env python

"talk to the finetuned model"

import time
from transformers import AutoConfig, AutoTokenizer, AutoModel, pipeline, GPT2LMHeadModel, GenerationConfig
from train import PROMPT_DICT

model_dir = 'output'

config = AutoConfig.from_pretrained( model_dir )
tokenizer = AutoTokenizer.from_pretrained( model_dir, cache_dir = model_dir, 
	max_length = config.n_positions, truncation = True )
model = GPT2LMHeadModel.from_pretrained( model_dir, config = config, cache_dir = model_dir )

generation_config = GenerationConfig(
    max_new_tokens = 100, # do_sample = True, top_k = 50, 
    eos_token_id = model.config.eos_token_id,
    pad_token_id = model.config.eos_token_id
)

nlp = pipeline( 'text-generation', model = model, tokenizer = tokenizer, generation_config = generation_config )


use_history = False

def toggle_history():
	global use_history
	use_history = not use_history

history = ''

"""
prompt = (
	"Below is an instruction that describes a task. "
	"Write a response that appropriately completes the request.\n\n"
	"### Instruction:\n{instruction}\n\n### Response:"
)
"""

prompt = PROMPT_DICT['prompt_no_input']
split_i = prompt.find( '###' )
history += prompt[:split_i]
prompt_postfix = prompt[split_i:]

while True:
	instruction = input( '\n> ' )
	
	if instruction == '.':
		toggle_history()
		if use_history:
			print( 'Prepending history to input.' )
		else:
			print( 'Not using history.' )
		continue
	
	if use_history:	
		input_txt = history + prompt_postfix.replace( '{instruction}', instruction )
	else:
		input_txt = prompt.replace( '{instruction}', instruction )
		
	#print( 'INPUT:', input_txt )
	
	t0 = time.time()
	result = nlp( input_txt )
	elapsed = time.time() - t0
	print( "({:.1f}s)".format( elapsed ))	
	
	generated_output = result[0]["generated_text"][len(input_txt):]
	print( generated_output )	
	
	history += prompt_postfix.replace( '{instruction}', instruction )
	history += generated_output + '\n\n'

