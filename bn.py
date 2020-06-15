import numpy as np
from collections import deque

class Neuron:
    
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
        return 1.0 if value > 0 else -1.0
    
    def learn(self,final_state, error_vals, error_mask):
        if not np.all(error_mask[self.outputs]):
            return False
        if len(self.inputs) > 0:
            inputs = final_state[self.inputs]
            error_vals[self.inputs] += self.weights*error_vals[self.index]
            return True
        
    def update(self,final_state,error_vals,scale,noise=0.1):
        if len(self.inputs) > 0:
            delta = error_vals[self.index]*final_state[self.inputs]*scale
            self.weights += np.random.normal(delta,np.abs(delta)*noise)
            delta = error_vals[self.index]*scale
            self.threshold += np.random.normal(delta,np.abs(delta)*noise)
        
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
    
class System:
    
    def __init__(self):
        self.neurons = []
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
    
    def finalize(self):
        outputs = [[] for neuron in self.neurons]
        for neuron in self.neurons:
            neuron.finalize()
            for input in neuron.inputs:
                outputs[input].append(neuron.index)
        for neuron in self.neurons:
            neuron.outputs = np.asarray(outputs[neuron.index],dtype=np.int32)
            neuron.outputs_set = frozenset(outputs[neuron.index])
        
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
    
    
    def learn(self,final_state,truth_vals,scale=0.5):
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
            neuron.update(final_state,error_vals,scale)
        