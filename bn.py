import numpy as np
from collections import deque

class Neuron:
    '''A neuron that implements a step activation function and accompanying error calculations.'''
    
    def __init__(self,index=None):
        self.index = index
        self.inputs = []
        self.outputs = None
        self.outputs_set = None
        self.weights = []
        self.threshold = 0
        
    def activate(self,state_current,only_inputs=False):
        if only_inputs:
            value = np.dot(self.weights,state_current) + self.threshold
        else:
            value = np.dot(self.weights,state_current[self.inputs]) + self.threshold
        return 1.0 if value > 0 else -1.0 #differentiate this!
    
    def learn(self,final_state, error_vals, error_mask):
        if not np.all(error_mask[self.outputs]):
            return False
        if len(self.inputs) > 0:
            inputs = final_state[self.inputs]
            error_vals[self.inputs] += self.weights*error_vals[self.index]
            return True
        
    def update(self,final_state,error_vals,scale,noise=None):
        if len(self.inputs) > 0:
            delta = error_vals[self.index]*final_state[self.inputs]*scale #no derivatives here!
            self.weights += np.random.normal(delta,np.abs(delta)*noise) if noise is not None else delta
            delta = error_vals[self.index]*scale #no derivatives here!
            self.threshold += np.random.normal(delta,np.abs(delta)*noise) if noise is not None else delta
        
    def add_input(self,index,weight=None):
        if weight is None:
            weight = np.random.normal(0.0,0.34)
        if type(index) is Neuron:
            index = index.index
        if index in self.inputs:
            index = self.inputs.index(index)
            self.weights[index] += weight
        else:
            self.inputs.append(index)
            self.weights.append(weight)
        
    def rem_input(self,index,weight):
        if type(index) is Nueron:
            index = index.index
        i = self.inputs.index(index)
        del self.inputs[i]
        del self.weights[i]
    
    def finalize(self):
        self.weights = np.asarray(self.weights,dtype=np.float32)
        self.inputs = np.asarray(self.inputs,dtype=np.int32)
    
    def __str__(self):
        if len(self.inputs) > 0:
            inputs = ['(N[%i] * %0.4f)'%(input,weight) for input,weight in zip(self.inputs,self.weights)]
            inputs = ' + '.join(inputs)
            return 'N[%i] = A[%s + %0.4f]'%(self.index,inputs,self.threshold)
        else:
            return 'N[%i] = input'%self.index
    
class Instance:
    '''Represents a structure built into a network'''
    def __init__(self,parent,output_neurons,output_shape):
        self.parent = parent
        self.neurons = output_neurons
        self.outputs = np.asarray([n.index for n in output_neurons],dtype=np.int32).reshape(output_shape)
        
    def get_state(self,state):
        return state[self.outputs].reshape(self.outputs.shape)
    
class Structure:
    '''Represents collections of neurons as an input->output device'''
    def construct(self,system,inputs):
        return np.asarray([],dtype=np.int32)
    def constrain(self,system,data):
        pass

class Input(Structure):
    '''Structure to inject values into the network'''
    def __init__(self,shape):
        self.shape = shape
        
    def construct(self,system,inputs):
        layer = [system.add_neuron(input=True) for i_inner in range(np.prod(self.shape,dtype=np.int32))]
        return Instance(self,layer,self.shape)
    
class Output(Structure):
    '''Structure to extract values from network'''
    def __init__(self,shape):
        self.shape = shape
        
    def construct(self,system,inputs):
        inputs = inputs.flatten()
        layer = [system.add_neuron(output=True) for i_inner in range(np.prod(self.shape,dtype=np.int32))]
        for neuron in layer:
            for i_input in inputs:
                neuron.add_input(i_input)
        return Instance(self,layer,self.shape)
        
class Dense(Structure):
    '''Structure that connects all neurons in the specified shape to all neurons of any shaped input'''
    def __init__(self,shape):
        self.shape = shape
        
    def construct(self,system,inputs):
        inputs = inputs.flatten()
        layer = [system.add_neuron() for i_inner in range(np.prod(self.shape,dtype=np.int32))]
        for neuron in layer:
            for i_input in inputs:
                neuron.add_input(i_input)
        outputs = np.asarray([n.index for n in layer],dtype=np.int32).reshape(self.shape)
        return Instance(self,layer,self.shape)
    
class Conv(Structure):
    '''Convolves an input shape with some dense kernel of neurons.
       Can use multiple kernels to add a dimension to the output if out_shape is specified.
       Works with a surprising variety of input, kernel, and output shapes.'''
    
    def __init__(self,kernel_shape=(3,3),out_shape=()):
        self.kernel_shape = kernel_shape
        self.out_shape = out_shape
        self.out_size = np.prod(self.out_shape,dtype=np.int32)
        
    def construct(self,system,inputs):
        conv_shape = tuple([in_dim-kernel_dim+1 for in_dim,kernel_dim in zip(inputs.shape,self.kernel_shape)])
        layer = [system.add_neuron() for i_inner in range(np.prod(conv_shape,dtype=np.int32)*self.out_size)]
        for i,c_index in enumerate(np.ndindex(*conv_shape)):
            for k_index in np.ndindex(*self.kernel_shape):
                u_index = tuple(np.asarray(c_index) + np.asarray(k_index)) #this is magical
                if len(self.out_shape) > 0:
                    if len(inputs.shape)-len(self.kernel_shape) > 0:
                        for j in range(i*self.out_size,(i+1)*self.out_size):
                            for input in inputs[u_index].flatten():
                                layer[j].add_input(input)
                    else:
                        for j in range(i*self.out_size,(i+1)*self.out_size):
                            layer[j].add_input(inputs[u_index])
                else:
                    if len(inputs.shape)-len(self.kernel_shape) > 0:
                        for input in inputs[u_index].flatten():
                            layer[i].add_input(input)
                    else:
                        layer[i].add_input(inputs[u_index])
                    
        system.add_constraint(self,layer)
        return Instance(self,layer,conv_shape+self.out_shape)
        
    def constrain(self,system,conv_neurons):
        for i in range(self.out_size):
            norm = len(conv_neurons)/self.out_size
            weights = np.sum([conv_neurons[j].weights for j in range(i,len(conv_neurons),self.out_size)],axis=0)/norm
            threshold = np.sum([conv_neurons[j].threshold for j in range(i,len(conv_neurons),self.out_size)],axis=0)/norm
            for j in range(i,len(conv_neurons),self.out_size):
                conv_neurons[j].weights[:] = weights
                conv_neurons[j].threshold = threshold
    
class System:
    '''Manages a neural network. Create neurons in the network with add_neuron. 
       Structures will call add_neuron when their construct methods are called.
       Contains logic for forward and back propagation, where neurons calculate their own errors.'''
    
    def __init__(self):
        self.neurons = []
        self.constraints = []
        self.inputs = []
        self.outputs = []
        self.recompute_cache = {}
        
    def add_neuron(self,neuron=None,input=False,output=False):
        if neuron is None:
            neuron = Neuron(index=len(self.neurons))
        else:
            neuron.index = len(self.neurons)
        self.neurons.append(neuron)
        if input:
            self.inputs.append(neuron.index)
        if output:
            self.outputs.append(neuron.index)
        return neuron
    
    def get_neuron(self,index):
        return self.neurons[index]
    
    def add_constraint(self,constraint,data):
        self.constraints.append((constraint,data))
    
    def finalize(self):
        outputs = [[] for neuron in self.neurons]
        for neuron in self.neurons:
            neuron.finalize()
            for input in neuron.inputs:
                outputs[input].append(neuron.index)
        for neuron in self.neurons:
            neuron.outputs = np.asarray(outputs[neuron.index],dtype=np.int32)
            neuron.outputs_set = frozenset(outputs[neuron.index])
        for constraint,data in self.constraints:
            constraint.constrain(constraint,data)
        
    def _empty_state(self):
        return np.zeros_like(self.neurons,dtype=np.bool),np.zeros_like(self.neurons,dtype=np.float32)
    
    def _step(self,state_changed,state):
        changed_indexes = np.nonzero(state_changed)[0]
        state_next_changed = np.zeros_like(state_changed)
        state_next = state.copy()
        if len(changed_indexes)/len(self.neurons) > 0.25: 
            #brute force stepping algorithm
            for index,neuron in enumerate(self.neurons):
                if np.any(np.in1d(neuron.inputs,changed_indexes,assume_unique=True)):
                    value = neuron.activate(state)
                    if value != state[neuron.index]:
                        state_next_changed[index] = True
                        state_next[index] = value
            return state_next_changed,state_next
        else: 
            #sparse stepping algorithm
            changed_indexes = frozenset(changed_indexes)
            if changed_indexes in self.recompute_cache:
                recompute_indexes = self.recompute_cache[changed_indexes]
            else:
                recompute_indexes = set()
                for i in changed_indexes:
                    recompute_indexes.update(self.neurons[i].outputs_set)
                #print('caching',changed_indexes,recompute_indexes)
                self.recompute_cache[changed_indexes] = frozenset(recompute_indexes)
            for index in recompute_indexes:
                neuron = self.neurons[index]
                value = neuron.activate(state)
                if value != state[neuron.index]:
                    state_next_changed[index] = True
                    state_next[index] = value
            return state_next_changed,state_next
    
    def guess(self,inputs,return_state=False):
        #for n in self.neurons:
        #    print(n)
            
        changed,state = self._empty_state()
        changed[self.inputs] = True
        state[self.inputs] = inputs
        
        #print('in',state[self.inputs])
        #print('start',state,changed)
        steps = 0
        while np.count_nonzero(changed) > 0:
            changed,state = self._step(changed,state)
            steps = steps + 1
            #print('step',state,changed)
        #print('out',state[self.outputs])
        #print('steps',steps)
        if return_state:
            return state[self.outputs],state
        else:
            return state[self.outputs]
    
    
    def learn(self,final_state,truth_vals,scale=1.0,noise=None):
        error_vals = np.zeros_like(final_state,dtype=np.float32)
        error_mask = np.zeros_like(final_state,dtype=np.bool)
        error_vals[self.outputs] = truth_vals - final_state[self.outputs]
        stack = deque(self.outputs)
        while len(stack) > 0:
            neuron = self.neurons[stack.popleft()]
            if neuron.learn(final_state,error_vals,error_mask):
                error_mask[neuron.index] = True
                for input in neuron.inputs:
                    if input not in stack:
                        stack.append(input)
        #print('errors',error_vals)
        for neuron in self.neurons:
            neuron.update(final_state,error_vals,scale,noise=noise)
        for constraint,data in self.constraints:
            constraint.constrain(constraint,data)
        