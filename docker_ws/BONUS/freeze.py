import argparse
from tensorflow.python.tools import freeze_graph

###############################################################################################################

def main():

	# Create arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('-i',  '--input_graph', 	  type=str,   default='./build/chkpts/inference_graph.pb',  help="Input graph binary protobuf file. Default is './build/chkpts/inference_graph.pb'.")
	parser.add_argument('-c',  '--input_checkpoint',  type=str,   default='./build/chkpts/float_model.ckpt',    help="Input checkpoint file. Default is './build/chkpts/float_model.ckpt'.")
	parser.add_argument('-n',  '--output_node_name',  type=str,   default='conv2d_3/BiasAdd', 		    help="Name of the graph's output node. Default is 'conv2d_3/BiasAdd'.")
	parser.add_argument('-o',  '--output_graph',      type=str,   default='./build/freeze/frozen_graph.pb',     help="Output (frozen) graph binary protobuf file. Default is './build/freeze/frozen_graph.pb'.")	

	# Parse arguments
	args = parser.parse_args()  

	# Print argument values
	print('\n------------------------------------')
	print ('Command line options:')
	print(' --input_graph:',      args.input_graph)
	print(' --input_checkpoint:', args.input_checkpoint)
	print(' --output_node_name:', args.output_node_name)
	print(' --output_graph:',     args.output_graph)
	print('------------------------------------\n')

	# Freeze graph
	freeze_graph.freeze_graph(input_graph=args.input_graph, input_checkpoint=args.input_checkpoint, input_binary=True, output_graph=args.output_graph, output_node_names=args.output_node_name, input_saver=None, restore_op_name=None, filename_tensor_name=None, clear_devices=True, initializer_nodes='') 

	# Log results
	print("Frozen graph", args.output_graph, "successfully created.\n")


###############################################################################################################

if __name__ == '__main__':
  main()

