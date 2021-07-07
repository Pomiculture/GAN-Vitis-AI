#!/bin/bash

###############################################################################################################

# Set all necessary environment variables used by the other scripts. 

###############################################################################################################

echo "Activating Anaconda Vitis AI TensorFlow..."  

# Activate the Conda virtual environment for Vitis AI with TensorFlow framework 
conda activate vitis-ai-tensorflow                                     

echo "Environment activated" 

###############################################################################################################

echo "Setting environment variables..." 

# Path to source directories
export SRC_AI_MODEL=./src/ai_model
export SRC_APP=./src/application
export SRC_PERFORMANCES=./src/performances
export SRC_DRIVE=./src/drive
export SRC_CALIB=./src/calibrate

###############################################################################################################

# Hardware
export ALVEO_MODEL=U280
export DPU_CONFIG=DPUCAHX8H # DPU version optimized for Alveo U280 
export ARCH=/opt/vitis_ai/compiler/arch/${DPU_CONFIG}/${ALVEO_MODEL}/arch.json     
export DPU_ARCHIVE=alveo_xclbin-1.3.0
export DPU_FREQ=14E300M                             

###############################################################################################################

# Set Output root directory
export RUNTIME_DIR=./RUNTIME

###############################################################################################################

# Root Data directory
export DATA_DIR=${RUNTIME_DIR}/data

# Set Data folder tree
export OUTPUT_DIR=${DATA_DIR}/output
export GOLD_DIR=${DATA_DIR}/gold
export SAMPLE_DIR=${DATA_DIR}/sample

# Set Drive config
export DRIVE_PARENT_ID=1peH5eJvFPtHqbAOO2cIlG8kn4zLquBIw
export DRIVE_FOLDER=Dataset
export CRED_PATH=${SRC_DRIVE}/client_secrets.json

###############################################################################################################

# Root Build directory
export BUILD=${RUNTIME_DIR}/build

# Set Build folder tree           
export TRAIN_DIR=${BUILD}/train
export FREEZE=${BUILD}/freeze
export QUANT=${BUILD}/quantize                
export COMPILE=${BUILD}/compile_${ALVEO_MODEL}
export TARGET=${BUILD}/target_${ALVEO_MODEL}
export MODEL_DIR=model_dir
export LOG=${BUILD}/logs

###############################################################################################################

# Checkpoints & Graphs filenames
export MODEL_FILENAME=generator.h5
export FROZEN_GRAPH_FILENAME=frozen_graph.pb

# Set log file names
export TRAIN_LOG=train.log
export FREEZE_LOG=freeze.log                                 
export QUANT_LOG=quant.log
export COMP_LOG=compile_${ALVEO_MODEL}.log                                                                              
export RUNTIME_LOG=runtime.log  
export EVAL_LOG=eval.log  
export DRIVE_LOG=drive.log                                                                

###############################################################################################################

# Network parameters
export NET_NAME=gan_generator
export KERAS_MODEL=${TRAIN_DIR}/${MODEL_FILENAME}
export SEED=42

# Input config
export CODINGS_SIZE=75
export INPUT_SHAPE=?,${CODINGS_SIZE},1,1
export INPUT_NODE_NAME=reshape_input  
   
# Output config
export OUTPUT_HEIGHT=28                                                     
export OUTPUT_WIDTH=28                              
export OUTPUT_NODE_NAME=reshape_1/Reshape                                

# Training parameters
export EPOCHS=40                                                          
export BATCH_SIZE=100                                                       

###############################################################################################################

# Quantization parameters
export NB_ITER=100
export CALIB_BATCH_SIZE=10

###############################################################################################################

# Runtime parameters
export NB_IMAGES=50
export IMG_FORMAT=png
export NB_THREADS=1

###############################################################################################################

# Evaluation parameters
export DISCRIMINATOR_NAME=discriminator.h5
export KERAS_DISCRIMINATOR=${TRAIN_DIR}/${DISCRIMINATOR_NAME}

###############################################################################################################

# Filter out TensorFlow INFO and WARNING logs (only get ERROR logs)
export TF_CPP_MIN_LOG_LEVEL=3
# Turn on memory growth to allocate only as much GPU memory as needed for the runtime allocations
export TF_FORCE_GPU_ALLOW_GROWTH=true

###############################################################################################################

# CUDA device order
export CUDA_DEVICE_ORDER=PCI_BUS_ID  
# List of GPUs to use (CUDA picks the fastest device as device 0)
export CUDA_VISIBLE_DEVICES=0                                                                     
