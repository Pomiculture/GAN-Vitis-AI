# Guide
This ressource explains the role of each file involved in the process.
The project consists in running different [bash scripts](https://ryanstutorials.net/bash-scripting-tutorial/bash-script.php "Bash") in a specific order.
The bahs files are located in the */workflow/* subfolder, while the Python source code is situated in the */src/* subfolder.
Open a terminal and make sure to be placed in the workspace directory **docker_ws**.
```
cd docker_ws
```
### Table of contents
- [Load the Vitis AI Docker environment](#docker)
- [Initialize the project](#init)
- [Build, train and run the AI model](#train)
- [Convert the Keras model to TensorFlow and freeze the graph](#freeze)
- [Quantize the model](#quantize)
- [Compile the model](#compile)
- [Prepare and run the application](#app)
- [Evaluate the results](#eval)
- [Export the results](#drive)
---
<div id='docker'/>

## 1) Load the Vitis AI Docker environment
The first step is to open the Vitis AI Docker image to access the Vitis AI tools and libraries.
We load the Docker image for CPU host : [xilinx/vitis-ai-cpu:latest](https://hub.docker.com/r/xilinx/vitis-ai-cpu "Docker Vitis AI CPU").
However, if you have a compatible NVIDIA graphics card with CUDA support, you can use GPU recipe.
We took the [script ```docker_run.sh``` from Xilinx](https://github.com/Xilinx/Vitis-AI/blob/master/docker_run.sh "Docker run")
and modified the location of the workspace to fit the structure of our project, and also [added the argument](https://github.com/Xilinx/Vitis-AI/issues/448 "Issue with Vitis AI Profiler on Docker") ```-v /sys/kernel/debug:/sys/kernel/debug  --privileged=true``` when calling the ```docker run``` command.
```
source ./workflow/0_run_docker_cpu.sh
```
---
<div id='init'/>

## 2) Initialize the project
Once we access the Docker virtual environment, we need to specify a few environment variables that will help to organize our project.
We also activate the Conda environment for the Vitis AI TensorFlow framework so as to use the proper Python libraries and Vitis AI commands during the process.
```
source ./workflow/1_set_env.sh
```
We can now build (or reset) the output directory that we called *RUNTIME*. It will contain the images produced by the GAN (root folder */RUNTIME/build*) and the different transformations of the model during the process (root folder */RUNTIME/data*).
```
source ./workflow/2_reset_output.sh
```
---
<div id='train'/>

## 3) Build, train and run the AI model
This step corresponds to the work of any software data scientist, meaning that we specify and train the [GAN model](https://github.com/ageron/handson-ml2/blob/master/17_autoencoders_and_gans.ipynb "GAN tutorial") using the [TensorFlow-Keras library](https://www.tensorflow.org/api_docs/python/tf/keras "tf.keras submodule") in [Python](https://www.python.org/ "Python").
```
source ./workflow/3_train_model.sh
```
This script calls the Python program *train_gan.py* located in the folder */src/ai_model*. We have to indicate the name of the generator and discriminator models we want to synthesize, the path to the output folder that wil contain these Keras models. For the training part, we specify the number of training iterations (epochs) and the size of the batches. We also fix a value for the seed that determines the random state so at to control it, get similar results at each run. Finally, we shape the network by indicating the size of the input noise matrix (codings_size) for the generator's input layer, and the width and height of the dataset images to define the shape of the generator's output layer and discriminator's input layer.

After training, we export the *generator* and *discriminator* using the Keras format ('.h5') to the folder */RUNTIME/build/train*. 
You can either check the logs  directly in the terminal or in the log file */RUNTIME/build/logs/train.log*.

To check the performances of the Keras generator, we run the following script that calls the Python one *run_model.py* from the folder */src/ai_model*. It takes as arguments the path to the keras model, that is to say */RUNTIME/build/train/generator.h5*, the seed value form randomness, but also the number of images to generate, their format (for instance '.png'), and the path to the folder that will contain these produced images, in our case */RUNTIME/data/gold*. We call it *gold* because we are now running the "original" model on the host machine, so it will serve as a reference when evaluating the transformed model to be run on the Alveo accelerator card.
```
source ./workflow/4_run_keras_model.sh
```
You can either check the logs  directly in the terminal or in the log file */RUNTIME/build/logs/runtime.log*.

---
<div id='freeze'/>

## 4) Convert the Keras model to TensorFlow and freeze the graph
Now that we have our basis, we can start the Vitis AI series of transformations.
Since we chose to use the Vitis AI TensorFlow framework, and since we are only interested in the model of the generator (now that it can produce images), we convert it to a TensorFlow graph ('.pb') and checkpoint file ('.ckpt'). After this step, we [freeze the inference graph](https://www.xilinx.com/html_docs/xilinx2019_2/vitis_doc/tensorflow_1x.html#uzv1576124117693__section_l3m_s3h_1kb "Getting the Frozen Inference Graph") in order to combine both graph and checkpoint files into a single binary protobuf file (.pb) and convert all the variables into constants so the weights don’t update during the quantization process. Furthermore, the graph nodes associated with training such as the optimizer and loss functions are stripped out.

We used to do these two steps separately, but we finally opted for a solution that processes both to get better results. To achieve that, we call an external GitHub project, [TF Keras YOLOv4/v3/v2 Modelset by David8862](https://github.com/david8862/keras-YOLOv3-model-set "TF Keras YOLOv4/v3/v2 Modelset") that contains a Python script that can convert a Keras model to a TensorFlow frozen graph. We just have to define the path to the Keras model, */RUNTIME/build/train/generator.h5*, and the path to the output frozen graph, let it be */RUNTIME/build/freeze/frozen_graph.pb*.
```
source ./workflow/5_keras_to_frozen_tf.sh
```
You can either check the logs  directly in the terminal or in the log file */RUNTIME/build/logs/freeze.log*.

In order to make it easier for the user, given that we need to identify the name of the input and output tensors of the frozen graph to proceed our transformations, we can run the script below that calls the Python program *get_io_tensors.py* located in the folder */src/ai_model*. We just have to specify the path to the frozen graph, that is to say */RUNTIME/build/freeze/frozen_graph.pb*. This script, inspired by the [NEWBEDEV website](https://newbedev.com/given-a-tensor-flow-model-graph-how-to-find-the-input-node-and-output-node-names "Given a tensor flow model graph, how to find the input node and output node names"), displays the name of the input and output tensors, and the shape of the input tensor.
```
source ./workflow/6_get_io_tensors.sh
```
In case you would like to visualize the whole architecture of the TensorFlow graph, you can run the following script, that uses TensorBoard. It calls the Python program *open_tensorboard.py* from the folder */src/ai_model*, and takes as input the frozen graph */RUNTIME/build/freeze/frozen_graph.pb*, a port number for TensorBoard, and a log directory, also for TensorBoard purposes. We ask you to open a link via a web browser to load the graph. You will then be able to quit by entering *CTRL-C* in the terminal.
```
source ./workflow/7_run_tensorboard.sh
```
---
<div id='quantize'/>

## 5) Quantize the model
We launch the [Vitis™ AI Quantizer for TensorFlow](https://www.xilinx.com/html_docs/xilinx2019_2/vitis_doc/tensorflow_1x.html#zuc1592307653938 "Vitis AI Quantizer") to convert the floating-point frozen graph */RUNTIME/build/freeze/frozen_graph.pb* (32-bit floating-point weights and activation values) to a fixed-point integer (8-bit integer - INT8) model by [quantizing](https://www.dictionary.com/browse/quantized "quantize") the weights/biases and activations to the given bit width. The output is the quantized model, which is saved to the path */RUNTIME/build/quantize/quantize_eval_model.pb*. After calibration, the quantized model is transformed into a DPU deployable model ready to be compiled.

We have to indicate the name of the input and output nodes from the frozen graph, and the input shape. We also specify a number of iterations used for the calibration part.

This process includes a calibration phase in which we use a callback function defined in the Python script *input_fn.py* from the folder */src/calibrate*. The callback function is named *calib_input* is run at each iteration of the calibration process. It creates a noise matrix, the same way as during the training process or when running the model, and we feed the input tensor with this batch of input data. We change the seed according to the iteration index to get new values at each iteration. 
```
source ./workflow/8_quantize_model.sh
```
You can either check the logs  directly in the terminal or in the log file */RUNTIME/build/logs/quant.log*.

To check whether the generator lost in quality, we can run the script below that runs the quantized graph */RUNTIME/build/quantize/quantize_eval_model.pb*, using the same pre-processing and post-processing than when running the Keras model. the produced images can be found in the folder */RUNTIME/data/quantize*. Again, we specify the number of images to generate, their format, the seed value, and this time we precise the name of the input and output nodes to extract them from the TensorFlow graph. This bash file calls a Python program named *run_graph.py* situated in the folder */src/ai_model/*.
```
source ./workflow/9_run_graph.sh
```
You can either check the logs  directly in the terminal or in the log file */RUNTIME/build/logs/runtime.log*.

---
<div id='compile'/>

## 6) Compile the model
The [Vitis™ AI Compiler for TensorFlow](https://www.xilinx.com/html_docs/xilinx2019_2/vitis_doc/compiling_model.html "Compiling the Model") is called by the following script to compile the model */RUNTIME/build/quantize/quantize_eval_model.pb* by mapping the network to an optimized DPU instruction sequence, according to the DPU of the target platform (DPUCAHX8H in our case). You can find the DPU configuration file in the path */opt/vitis_ai/compiler/arch/DPUCAHX8H/U280/arch.json*. The XCompiler (XIR based Compiler) constructs an internal computation graph as intermediate representation (IR) and performs several optimization steps.

We provide a name for the compiled model, *gan_generator.xmodel*, and we place it into the output folder */RUNTIME/build/compile_U280/*.
```
source ./workflow/10_compile_model.sh
```
You can either check the logs  directly in the terminal or in the log file */RUNTIME/build/logs/compile_U280.log*.

---
<div id='app'/>

## 7) Prepare and run the application
The application code ...

...

To facilitate the runtime process, we gather the application code from */src/application/* and compiled model */RUNTIME/build/compile_U280/gan_generator.xmodel* to a same folder */RUNTIME/build/target_U280/*.
```
source ./workflow/11_make_target.sh
```
We can now load the overlay that suits the DPU of type DPUCAHX8H so as to configure the accelerator card. We have to download an archive, *alveo_xclbin*, untar it, and copy the files for the target platform, *U280/14E300M/dpu.xclbin* and *U280/14E300M/hbm_address_assignment.txt*, to the */usr/lib* folder.
We only download the archive if it is not already downloaded to the workspace.
```
source ./workflow/12_load_u280_overlay.sh
```
To run the application, run the following script.
```
source ./workflow/13_run_app.sh
```
You can either check the logs  directly in the terminal or in the log file */RUNTIME/build/logs/runtime.log*.

---
<div id='eval'/>

## 8) Evaluate the results
...

---
<div id='drive'/>

## 9) Export the results
This optional step consists in exporting the output images to a Google Drive folder.
```
source ./workflow/15_export_results.sh
```

---
TODO : Export results - blabla libraries + how to GDrive proper client secret et changer dans src + src file
TODO : expliquer les 3 modes d'évaluation et readme à part pour expliquer et présenter images + eval log + call the script 14
TODO : APP explanations
expliquer code app + run_app
/usr/bin/python3
Python API (link + voir explications du code + expliquer les main steps)
https://www.xilinx.com/html_docs/vitis_ai/1_3/compiling_model.html#ztl1570696058091
--model 	${TARGET}/${MODEL_DIR}/${NET_NAME}.xmodel \
		--threads 	${NB_THREADS} \
		--output_folder ${OUTPUT_DIR} \
		--format 	${IMG_FORMAT} \
		--num_images 	${NB_IMAGES} \
		--seed 		${SEED} \
		--codings_size 	${CODINGS_SIZE} 
TODO : expliquer les 3 méthoes d'éval (et cf. Pwpt fiches)
TODO : https://www.xilinx.com/html_docs/vitis_ai/1_3/eku1570695929094.html
TODO : picture quantize + picture compile
TODO : remove in ash file the "blabla set of images"
+ check pwpt fiches de code + caser figure schéma custom
	uet où regarder et quoi changer si nouveau (setenv, dataset)
	low bandwidth
| Luc| Lucos|  
|:---: | :---:  |
| 1  |  2   |
| 3  |  4   |  
| 5  |  6   |
+ TODO : screen TensorBoard
+ TODO : show output images and fps (and score) at each running step *3
+ parler de file_management
+ parler de dpu_runner
+ schéma du process Vitis AI workflow
+ Faire folder tree pour output folder RUNTIME après complete process
+ + TREE pour workflow and src Whole project without runtime output dir
steps : https://www.xilinx.com/products/design-tools/vitis/vitis-ai.html
For Drive export : create your own app (cf. link) on Google Drive (own client_secret file) to connect to your private Gogole dRive space

