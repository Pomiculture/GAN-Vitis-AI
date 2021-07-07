import os
import sys
import argparse

import tensorflow as tf 
import tensorflow.python.keras.backend as keras_backend
from tensorflow.keras.models import load_model

###############################################################################################################

def keras2tf(keras_hdf5_path, tf_ckpt_path, tf_graph_path):
    '''
    Write out checkpoint & inference graph for use with TensorFlow "freeze_graph" script, from Keras HDF5 model.
    '''

    # Set learning phase for no training (indicate to layers such as dropout or batch normalization that we are no longer training)
    keras_backend.set_learning_phase(0)

    # Load Keras model
    loaded_model = load_model(keras_hdf5_path)

    # Write out the input and output tensor names as environment variables
    #os.environ['INPUT_NODE_NAME'] = loaded_model.inputs[0].name
    #os.environ['OUTPUT_NODE_NAME'] = loaded_model.outputs[0].name

    # Display Keras model info
    print('-------------------------------------')
    print ('Keras model information:')
    print ('- Input node name:', loaded_model.inputs[0].name)
    print ('- Output node name:', loaded_model.outputs[0].name)
    print('-------------------------------------')

    # Print the CNN structure
    print(loaded_model.summary())

    # Fetch the Tensorflow session using the Keras backend
    session = keras_backend.get_session()

    # Get the Tensorflow session graph
    input_graph_def = session.graph.as_graph_def()

    # Save checkpoint
    #saver = tf.compat.v1.train.Saver()
    #saver.save(session, tf_ckpt_path)
    tf.compat.v1.saved_model.simple_save(session, tf_ckpt_path)

    # Write out inference graph 
    graph_head_tail = os.path.split(tf_graph_path) 
    tf.io.write_graph(input_graph_def, graph_head_tail[0], graph_head_tail[1], as_text=False)

    # Display TensorFlow checkpoint info
    print ('TensorFlow information:')
    print ('- Checkpoint saved as:', tf_ckpt_path)
    print ('- Graph saved as:', tf_graph_path)
    print('-------------------------------------')

    return


###############################################################################################################

def main():

    # Create arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-k',  '--input_keras_hdf5',   type=str,   default='./model.hdf5',    help="Path to Keras HDF5 file. Default is './model.hdf5'.")
    parser.add_argument('-c',  '--output_tf_ckpt',     type=str,   default='./tf_ckpt.ckpt',  help="Path to Keras checkpoint file. Default is './tf_ckpt.ckpt'.")
    parser.add_argument('-g',  '--output_tf_graph',    type=str,   default='./tf_graph',      help="Path to inference graph file. Default is './tf_graph.ckpt'.")

    # Parse arguments
    args = parser.parse_args()  

    # Print argument values
    print('\n------------------------------------')
    print('TensorFlow version : ',tf.__version__)
    print(sys.version)
    print('------------------------------------')
    print ('Command line options:')
    print(' --keras_hdf5 (input):', args.input_keras_hdf5)
    print(' --tf_ckpt (output):',   args.output_tf_ckpt)
    print(' --tf_graph (output):',  args.output_tf_graph)
    print('------------------------------------\n')

    # Get TensorFlow checkpoint and inference graph from Keras HDF5 model
    keras2tf(args.input_keras_hdf5, args.output_tf_ckpt, args.output_tf_graph)


###############################################################################################################

if __name__ == '__main__':
    main()

