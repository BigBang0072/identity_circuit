import numpy as np
from  pprint import pprint
import scipy
import pdb

import matplotlib
import matplotlib.pyplot as plt
plt.figure(figsize=(15,8))

import qiskit
from qiskit import Aer
from qiskit.visualization import *

class IdentityBlock():
    '''
    This class will generate one layer of the combination
    of even and odd block to be applied to the circuit.
    '''

    def __init__(self,n_qubits,circuit,layer_num,sym_thetas):
        '''
        Parameters:
            n_qubits        : number of qubits in the circuit
            circuit         : the circuit defined so for
            layer_num       : the index of layer (for identifying params)
        '''
        self.circuit=circuit

        #Applying the approporate blocks to the circuit
        self._apply_odd_block(n_qubits,layer_num)
        self._apply_even_block(n_qubits,layer_num)

        self._collect_params(sym_thetas)       
        
    def _apply_odd_block(self,n_qubits,layer_num):
        '''
        This function will apply the odd block to the circuit as required
        '''
        #Initializing the parameters
        param_name="theta.odd.{}.{}"
        self.odd_thetas={param_name.format(layer_num,idx):
                            qiskit.circuit.Parameter(param_name.format(layer_num,idx))
                         for idx in range(n_qubits)
                        }
        
        #Now we will append the required gate to the circuit
        _ = [self.circuit.rx(self.odd_thetas[pname],idx) 
                for idx,pname in enumerate(self.odd_thetas.keys())]
        
    def _apply_even_block(self,n_qubits,layer_num):
        '''
        This function will apply the even circuit to the circuit as required
        '''
        #Initializing the parameters
        param_name="theta.even.{}.{}"
        self.even_thetas={param_name.format(layer_num,idx):
                            qiskit.circuit.Parameter(param_name.format(layer_num,idx))
                         for idx in range(n_qubits)
                        }
        #Now we will append the required gate to the circuit
        _ = [self.circuit.rz(self.even_thetas[pname],idx) 
                for idx,pname in enumerate(self.even_thetas.keys())]
        
        #Now we will apply the two qubit gate
        self.circuit.cz(control_qubit=0,target_qubit=1)
        self.circuit.cz(control_qubit=0,target_qubit=2)
        self.circuit.cz(control_qubit=0,target_qubit=3)

        self.circuit.cz(control_qubit=1,target_qubit=2)
        self.circuit.cz(control_qubit=1,target_qubit=3)

        self.circuit.cz(control_qubit=2,target_qubit=3)

    def _collect_params(self,sym_thetas):
        '''
        This function will collect the parameters defined in these bolck in one
        centralized location.
        '''
        for pname in self.odd_thetas.keys():
            sym_thetas[pname]=self.odd_thetas[pname]
        
        for pname in self.even_thetas.keys():
            sym_thetas[pname]=self.even_thetas[pname]

class IdentityCircuit():
    '''
    This class implements a variational circuit which we will optimize
    to produce an identity function. The parameters of this circuit will be 
    updated using gradient descent or grid search. 
    '''

    def __init__(self,n_qubits,init_state,n_layers,backend,shots=None):
        '''
        Parameters:
            init_state  : state to initialize the quantum circuit
            n_layers    : number of layers of identity block to be used
            backend     : to be used for running the simulation
            shots       : number of repetition/ measurement to get final state
        '''
        #First of all lets define the circuit
        assert n_qubits==4,"Qubit should be 4 as per the problem"
        self.n_qubits   =   n_qubits
        self.circuit    =   qiskit.QuantumCircuit(self.n_qubits)

        #Initialzing the circuit with the given random state
        self.init_state=init_state
        self.init_state_vec=self._encode_input_state()
        self.circuit.initialize(self.init_state_vec,list(range(self.n_qubits)))

        #Now lets apply the even and odd blocks
        self.n_layers=n_layers
        self.sym_thetas={}
        for layer_num in range(n_layers):
            self.circuit = IdentityBlock(self.n_qubits,self.circuit,layer_num,self.sym_thetas).circuit
        #Collecting the names of each thetas
        self.thetas_name=list(self.sym_thetas.keys())
        
        #Finally lets measure the circuit
        #self.circuit.measure_all()
    
        #Initialiing the backend to be used
        self.backend    =   backend
        self.shots      =   shots

        #Initializing the variables which will hold our metrics for viz
        self.all_cost=[]        #The loss value at each epoch
        self.step=0             #The epoch number
    
    def _encode_input_state(self,):
        '''
        This fucntion will encode the input state in little endian format
        for later comparison with final state.

        This function assumes that the input state is a pure state.
        '''
        #First of all we will calculate the index of all possible state
        entry_idx = sum([val*(2**idx) for idx,val in enumerate(self.init_state)])

        #Creating the init_state
        init_state_vec=np.zeros(2**self.n_qubits)
        init_state_vec[entry_idx]=1

        return init_state_vec
    
    def simulate(self,thetas):
        '''
        Parameters:
            thetas      : the dictionary of value parameters used in the circuit
        '''
        #Forward propagating through the circuit
        job = qiskit.execute(self.circuit,
                            self.backend,
                            shots=self.shots,
                            parameter_binds=[self._bind_parameters(thetas)])
        #Getting the measurement result
        output_state_vec = job.result().get_statevector(self.circuit)

        return output_state_vec
    
    def _bind_parameters(self,thetas):
        '''
        This function will create a paremeter binding for execution based
        on the type of theta given to it.

        If the theta is from our grad descent its in different format
        than the one sent by optimizer from qiskit
        '''
        if type(thetas)==type(dict()):
            return {
                    self.sym_thetas[pname]:val
                        for pname,val in thetas.items()
            }
        else:
            return {
                    self.sym_thetas[pname]:val
                        for pname,val in zip(self.thetas_name,list(thetas))
            }
    
    def calculate_cost(self,thetas):
        '''
        This function will calcualte the cost by 

            cost = <delta | delta>

            |delta> = pshi(theta) - phi (Does this make sense)?
        '''
        #Simulating the circuit to get the ourput vector
        output_state_vec=self.simulate(thetas)

        delta_vec=self.init_state_vec-output_state_vec
        cost = np.inner(delta_vec, np.conjugate(delta_vec))

        assert cost.imag==0.0,"Cost cannot be imaginary"
        cost=cost.real

        #Saving the metric for visualization
        self.all_cost.append(cost)
        if self.step%5==0:
            self._save_network_metrics(output_state_vec,thetas)
        self.step+=1
        
        return cost
    
    def _save_network_metrics(self,output_state_vec,thetas):
        '''
        This function will save the network metrics like the trend of
        parameters, and the final statevector.
        '''
        #Get the theta binding
        thetas=self._bind_parameters(thetas)
        theta_names=[ele.name for ele in thetas.keys()]
        #First of all lets plot the thetas
        ax1=plt.subplot(211)
        ax1.bar(range(1,len(thetas)+1),list(thetas.values()),color="b",
                label="value")
        ax1.set_title("Parameter/Theta Values at Epoch:{}".format(self.step))
        ax1.set_xticks(range(1,len(theta_names)+1))
        ax1.set_ylim([0,(2.2)*np.pi])
        # ax1.set_xticklabels(theta_names,rotation=45)
        # ax1.legend()
        

        #Now get the ral and imaginary part of the complex number
        imag_part=np.imag(output_state_vec)
        real_part=np.real(output_state_vec)
        state_names=["{0:04b}".format(num) for num in range(2**n_qubits)]
        #Now lets plot them
        ax2=plt.subplot(212)
        width=0.35
        index=np.arange((len(imag_part)))
        ax2.bar(index-width/2,real_part,width,
                color="g",label="real")
        ax2.bar(index+width/2,imag_part,width,
                color="r",label="imag")
        ax2.set_title("Coefficient of State Vector at Epoch:{}".format(self.step))
        ax2.set_xticks(range(0,len(state_names)))
        ax2.set_xticklabels(state_names,rotation=45)
        ax2.set_ylim([-1.5,1.5])
        ax2.legend(["real","imag"])

        plt.savefig("{}/step:{}.png".format(self.n_layers,self.step))
        # pdb.set_trace()
        plt.clf()

    def _calculate_gradient(self,thetas,epsilon):
        '''
        This fucntion will calcualte the numberical gradient of quantum circuit
        using the finite difference method

                grad = output(theta+epsilon) - output(theta-epsilon) / (2*epsilon)
        '''
        grads={}
        thetas_copy=thetas.copy()

        #One by one we will find gradient with respect to every variable
        for pname in thetas.keys():
            #Forward Pertubation on current theta
            thetas_copy[pname]=thetas[pname]+epsilon
            cost_plus=self.calculate_cost(thetas_copy)

            #Backward pertubation of current theta
            thetas_copy[pname]=thetas[pname]-epsilon
            cost_minus=self.calculate_cost(thetas_copy)

            #Resetting the theta state
            thetas_copy[pname]=thetas[pname]

            #Calculating the gradient
            grad=(cost_plus-cost_minus)/(2*epsilon)
            grads[pname]=grad
        
        return grads
    
    def _initialize_thetas(self,ret_type="dict"):
        '''
        This function will randomly initialize the parameters used in this
        network.
        '''
        thetas={}
        for layer_num in range(self.n_layers):
            for qidx in range(self.n_qubits):
                thetas["theta.odd.{}.{}".format(layer_num,qidx)]=np.random.uniform(0,2*np.pi)
                thetas["theta.even.{}.{}".format(layer_num,qidx)]=np.random.uniform(0,2*np.pi)

        if ret_type=="dict":
            return thetas
        #If we want the initial parametes as a list
        return list(thetas.values())

    def _gradient_descent_step(self,thetas,grads,lr):
        '''
        This function will apply a step of vanilla gradient descent
        '''
        for pname in thetas.keys():
            thetas[pname]=thetas[pname]-lr*grads[pname]
        
    def optimize(self,n_itr,epsilon=0.01,lr=0.001):
        '''
        This function will apply gradient descent on the cost function
        to get the parameters and make the whole circuit identity
        '''
        #Initialize the parameters
        thetas=self._initialize_thetas()
        #final_output_vector=None

        #Now we will start the optimization process
        for iidx in range(n_itr):
            #Finding the current cost of circuit
            cost=self.calculate_cost(thetas)
            output_state_vec=self.simulate(thetas)

            #Now computing the gradient and applying
            grads=self._calculate_gradient(thetas,epsilon)
            self._gradient_descent_step(thetas,grads,lr)

            print("Epoch:{}\t Cost:{}".format(iidx,cost))
            pprint(thetas)
            print("Output Vector:")
            pprint(output_state_vec)
            print("\n")
        
        print("Training Completed")
    
    def optimize_with_inbuilt_function(self,n_itr,tol):
        '''
        The gradient descent optimizer implemented above dosent put constraints
        on the variable. Hence we will use the inbuilt optimizer.
        '''
        print("Starting the Training!!")
        #Initialize the parameters
        thetas=self._initialize_thetas(ret_type="list")

        #Using inbuilt scipy function to minimize
        opt_results = scipy.optimize.minimize(self.calculate_cost,
                                            thetas,
                                            method="SLSQP",
                                            bounds=[(0, 2*np.pi)]*len(thetas))


        print("Training Completed!!")
        #Getting the final parameters
        final_thetas=list(opt_results.x)
        cost=self.calculate_cost(final_thetas)
        output_state_vec=self.simulate(final_thetas)
        final_thetas=self._bind_parameters(final_thetas)
        
        print("Final Loss:",cost)
        pprint(final_thetas)
        print("Output Vector:")
        pprint(output_state_vec)
        print("\n")


if __name__=="__main__":
    #Setting up the network parameters
    n_qubits=4
    #Creating a random initial state (Assuming pure state for now)
    init_state=[1,1,0,0]        # [q0,q1,q3,q3] -->Big Endien form right now

    n_layers=1
    backend=Aer.get_backend("statevector_simulator")

    #Now lets create the curcuit
    circuit=IdentityCircuit(n_qubits,init_state,n_layers,backend)

    #Now we will start the optimization process
    optimizer_type="inbuilt"    #[grad_descent,inbuilt]
    if optimizer_type=="grad_descent":
        epochs=500
        epsilon=0.01
        lr=0.1
        circuit.optimize(n_itr=epochs,
                            epsilon=epsilon,
                            lr=lr)
    else:
        #Optimizing the parameters using inbuilt optimizer
        epochs=500
        tol=1e-9
        circuit.optimize_with_inbuilt_function(n_itr=epochs,tol=tol)
    
    #Saving the loss trend in a text file
    save_fname="{}/loss.txt".format(n_layers)
    np.savetxt(save_fname,circuit.all_cost,delimiter="\t")
    
    
















