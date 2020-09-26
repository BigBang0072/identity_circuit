import numpy as np

import matplotlib
import matplotlib.pyplot as plt
# plt.figure(figsize=(15,8))

def load_loss_file(layer_num,base_path="logs/{}/loss.txt"):
    path = base_path.format(layer_num)
    return np.loadtxt(path,delimiter="\t")

def plot_comparative_trend(layers):
    '''
    Plot the loss curve of all the different layers in one single graph
    '''
    for layer in layers:
        #Load the loss data from the logs
        loss_arr=load_loss_file(layer)
        plt.plot(loss_arr,label="num_layer:{}".format(layer))
    
    plt.legend()
    plt.xlabel("Number of Epochs")
    plt.ylabel("Error Value")
    plt.title("Loss Curve with Layers in Circuit (tol:1e-9)")
    plt.show()

if __name__=="__main__":
    layers=[1,2,4,8,16]
    plot_comparative_trend(layers)