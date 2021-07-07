#%tensorflow_version 1.x
import tensorflow as tf
#tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
# Log name of device used for execution
#tf.debugging.set_log_device_placement(True)

import argparse
from tensorflow.python.client import device_lib
from enum import Enum
import timeit

import run_model

###############################################################################################################

class Mode(Enum):
  CPU = 'cpu'
  GPU = 'gpu'

  @classmethod
  def has_value(cls, value):
      return value in cls._value2member_map_ 


###############################################################################################################

def user_request():
    """
    Ask user to enter the desired device type to use.
    """
    mode = input('Enter the name of the device type: ') # or 'cpu'
    print(f'You entered: {mode}')
    return mode


###############################################################################################################

def get_devices():
    """
    Get the list of all available devices.
    """
    devices = device_lib.list_local_devices()
    if not devices:
        raise SystemError("No device was found.")
    return devices


def get_device_info(device):
    """
    Get 'device' info (id, memory limit, model name) according to its type.
    """
    print('\n------------------------------------')
    # Get device type
    print('Device type:', device.device_type)
    # Get device id
    print("Device name:", device.name)
    # Get memory limit
    print("Memory limit:", device.memory_limit, "bytes")
    # Get model name
    if(device.device_type == Mode('cpu').name):
        f = open("/proc/cpuinfo", "r")
        Lines = f.readlines()
        f.close()
        # Strips the newline character
        for line in Lines:
            if(line.find('model name') != -1):
                print(device.device_type, "model: [", line.strip(), "]")
                break
    elif(device.device_type == Mode('gpu').name):
        # Get device model
        device_model = [d for d in device.physical_device_desc.split(", ") if d.find("name") != -1] 
        print(device.device_type, "model:", device_model)
    print('------------------------------------')


def find_device(device_list, mode):
    """
    Select the first device of right type in list, if it exists.
    """
    found_device = False
    for device in device_list:
        if(device.device_type == Mode(mode).name):
            # Exit search
            found_device = True
            break

    # Check whether no device matches the type (e.g. GPU mode not enabled)
    if not found_device:
        err_msg = "Enable GPUs for the Google Colab notebook.\n Navigate to 'Edit' -> 'Notebootk Settings' and select the 'GPU' accelerator type in the dropdown list" 
        raise SystemError(err_msg)

    return device


###############################################################################################################

def get_ram_info():
    """
    Check the available memory resources (RAM).
    """
    f = open("/proc/meminfo", "r")
    text = f.read()
    #print(text)
    f.close()


###############################################################################################################

def runtime(device_name, path_to_model, output_folder, num_images, format, seed):
    """
    Execute application on device named 'device_name' and get execution time.
    """
    with tf.device(device_name):
        # Run process
        print("\nRunning application...") 
        print('------------------------------------')
        t = timeit.Timer(lambda: run_model.inference(path_to_model, output_folder, num_images, format, seed))  
        duration = t.timeit(number=1)  # Increment the 'number' of executions to get more reliable time measures
        print('------------------------------------')
        print("Execution time :", duration, "s")
        print('------------------------------------\n')
        return duration


def calculate_speedup(t_cpu=1, t_gpu=0.5):
    """
    Calculate speedup between GPU and CPU.
    """
    speedup = round(t_cpu/t_gpu, 5)
    print('GPU speedup factor over GPU:', speedup)
    return speedup


 ###############################################################################################################

def run(device_type, path_to_model, output_folder, num_images, format, seed):

    if not device_type:
        # Ask user to choose a device type (CPU or GPU)
        mode = user_request()
    else:
        mode = device_type

    # Check whether the device type is valid
    if not Mode.has_value(mode):
        raise ValueError("Please choose a valid device type among the following :", [m.value for m in Mode])

    # Get the total list of devices
    devices = get_devices()

    # Connect to the first device of right type in list
    device = find_device(devices, mode)

    # Get device info
    get_device_info(device)

    # Get RAM info
    get_ram_info()

    # Run App with device 
    return runtime(device.name, path_to_model, output_folder, num_images, format, seed)


###############################################################################################################

def main():

    # Create arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-d',  '--device_type',   type=str,   default='cpu',                                  help="Type of device used to run the application. Default is 'cpu'.")
    parser.add_argument('-p',  '--path_to_model', type=str,   default='./keras_h5/keras_generator_model.h5',  help="Path to Keras model of the GAN's generator. Default is './keras_h5/keras_generator_model.h5'.")
    parser.add_argument('-o',  '--output_folder', type=str,   default='./out',                                help="Folder to store the output images. Default is './out'.")
    parser.add_argument('-n',  '--num_images',    type=int,   default=8,                                      help="Number of images to produce. Default is '8'.")
    parser.add_argument('-f',  '--format',        type=str,   default='png',                                  help="Output image format. Default is 'png'.")
    parser.add_argument('-s',  '--seed',          type=int,   default=42,                                     help="Set seed value. Default is '42'.")

    # Parse arguments
    args = parser.parse_args()  

    # Print argument values
    print('\n------------------------------------')
    print ('Command line options:')
    print(' --device_type:',   args.device_type)
    print(' --path_to_model:', args.path_to_model)
    print(' --output_folder:', args.output_folder)
    print(' --num_images:',    args.num_images)
    print(' --format:',        args.format)
    print(' --seed:',          args.seed)
    print('------------------------------------\n')

    # Run the GAN's generator to produce images using a specific device
    return run(args.device_type, args.path_to_model, args.output_folder, args.num_images, args.format, args.seed)       


###############################################################################################################

if __name__ == '__main__':
    main()