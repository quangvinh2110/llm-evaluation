#!/bin/bash

python3 evaluate_llms_async.py \
--endpoint_ip "http://0.0.0.0:8264" \
--served_model_name "Vi-Llama-3.3-70B-Instruct" \
--model_path "/raid/HUB_LLM/010225_llama33_70b_instruct/checkpoint-1470" \
--engine "vllm" \
--dataset_name "vmlu" 
