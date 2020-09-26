import numpy as np

import matplotlib
import matplotlib.pyplot as plt
# plt.figure(figsize=(15,8))

def load_loss_file(path):
    return np.loadtxt(path,delimiter="\t")

def plot_comparative_trend(config_name,base_path):
    '''
    Plot the loss curve of all the different layers in one single graph
    '''
    for config in config_name:
        #Load the loss data from the logs
        loss_arr=load_loss_file(base_path+config)
        plt.plot(loss_arr,label="{}".format(config))
    
    plt.legend()
    plt.xlabel("Number of Epochs")
    plt.ylabel("Error Value")
    plt.title("Loss Curve with Layers in Circuit (tol:1e-9)")
    plt.show()

if __name__=="__main__":
    # layers=[1,2,4,8,16]
    config_name=[
                #"2.rx+rx.txt","2.rx+ry.txt","2.rx+rz.txt",
                #"2.ry+rx.txt","2.ry+ry.txt","2.ry+rz.txt",
                "2.rz+rx.txt","2.rz+ry.txt","2.rz+rz.txt",
                ]
    base_path="logs/diff_gate/"
    plot_comparative_trend(config_name,base_path)