#!/usr/bin/env python

import os
import modal

training_script = 'train.py'
model_name = 'gpt2'					# huggingface name
gpu_count = 1
modal_volume_name = "test_volume"

stub = modal.Stub( "AlpacaGPT" )			# this is a modal app name

@stub.function( 
	image = modal.Image.debian_slim().pip_install_from_requirements( 'requirements.txt' ),
	shared_volumes = { "/finetune": modal.SharedVolume.from_name( modal_volume_name )},
	mounts = [ 
		modal.Mount.from_local_file( training_script, remote_path = "/{}".format( training_script )),
		modal.Mount.from_local_dir( "data/modal", remote_path = "/data/modal" ), 
		] + modal.create_package_mounts([ "utils", ]),
	gpu = modal.gpu.A10G( count = gpu_count ),
	timeout = 60 * 60 *24,
)
def run_train():

	cmd = "python /{} \
		--model_name_or_path {} \
		\
		--cache_dir /finetune/cache \
		--data_path /data/modal/train.json \
		--output_dir /finetune/output \
		\
		--bf16 True \
		--num_train_epochs 3 \
		--per_device_train_batch_size 1 \
		--per_device_eval_batch_size 1 \
		--gradient_accumulation_steps 32 \
		--evaluation_strategy no \
		--save_strategy steps \
		--save_steps 2000 \
		--save_total_limit 1 \
		--learning_rate 2e-5 \
		--weight_decay 0. \
		--warmup_ratio 0.03 \
		--lr_scheduler_type cosine \
		--logging_steps 1 \
		--report_to none".format( training_script, model_name )
		
	os.system( cmd )

if __name__ == "__main__":
	with stub.run():
		run_train.call()


