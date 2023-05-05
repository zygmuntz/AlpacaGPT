# AlpacaGPT

How to train your own ChatGPT, Alpaca style

	convert_training_data.py - convert alpaca data from instruction-input-output to instruction-output format
	LICENSE.txt - original alpaca license
	modal_download_files.py - download files from modal after training
	modal_run.py - run train.py on modal using one GPU
	requirements.txt - original alpaca requirements to run train.py
	talk.py - talk to the finutuned model
	train.py - slightly modified original alpaca training script
	utils.py - trimmed original alpaca utils script
	
If you want to train on multiple GPUs, modify `modal_run.py` by setting the GPU count and copying FSDP parameters from the original alpaca repo.

To run it, you need to set up your modal account. Then create a shared volume on modal, for example

	modal volume create test_volume
	
Then inspect and run `modal_run.py`. After finetuning, download the files and possibly delete them from the remote drive:

	python modal_download_files.py
	modal volume rm -r test_volume /
