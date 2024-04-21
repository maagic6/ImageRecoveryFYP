#!/bin/bash

#install requirements
pip install -r requirements.txt

#set environment variables (amd)
#export HSA_OVERRIDE_GFX_VERSION=10.3.0
#export PYTORCH_ROCM_ARCH="gfx1030"

#run gradio script
python app_v1.py