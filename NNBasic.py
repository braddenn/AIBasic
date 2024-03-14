# Module: NNBasic.py
# Desc: AI for NNBasic.
#	For an input set (one of several input lists) compute the output set. 
#	There can be an output neuron that is trained for each input set.
#	This is a very basic Neural Net defined in parameter config
#	(#layers, #neurons/layer) as defined in a JSON configuration file.
# Structure:
# 	class NNBasic -  Neural Net AI for multiple inputs, returns a computed result
#	class Layer - a vertical layer of neurons or comparators. 
#   class Neuron - create a neuron, connect outputs to next layer, set/adjust weights
#
#	Derived from github code:
#		2015: https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
#			AI structure and all parameters defined in the config file.
#			Prints state after each step if DEBUG (in config) is True.
#			Saves all weights and reference to a match file after adaption.
#			Will match the input values iff (if and only if) it is started with the match file.
#
#	Comment references:
#	[1] Wikipedia article on Backpropagation
#		http://en.wikipedia.org/wiki/Backpropagation#Finding_the_derivative_of_the_error
#	[2] Neural Networks for Machine Learning course on Coursera by Geoffrey Hinton
#		https://class.coursera.org/neuralnets-2012-001/lecture/39
#	[3] The Back Propagation Algorithm
#		https://www4.rgu.ac.uk/files/chapter3%20-%20bp.pdf  
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
#   History:
#   	Derived from AiLab
#   	Author: Brad Denniston
#   	Version: 0.1, 15 Jan 2024
#

import random
import math
# import cmath --- drop if not needed

# CLASS: NeuralNetNBasic
# Desc: Neural Net AI for multiple inputs, returns a computed result.
#	The AI structure is defined by a JSON config file.
# Usage: used by user.py
class NeuralNetBasic:
	# DEF: __init__ def of class NeuralNetBasic
	# Desc: The structure of this neural net is defined in a passed in dictionary
	#
	# Parm: config -  dictionary of configuration data
	# 	"num_inputs" - define number of input signals/neurons
	# 	"num_layers" - count of signal layer, input, hidden and output layers
	#						external signals comprise the signal layer (0)
	#	"test set count" - 1 or more output signals, 1 per test set
	#	"learning rate" : somewhere around 0.001
	#	"cycle_limit" : stop regression when # cycles gets to this value
 	#	"error_limit" : stop regression when calculation error is this small
	#
	# Usage: public, started and managed in a main module
	def __init__(self, config ):
		self.config = config
		if self.config['debug'] == 'true' :
			self.DEBUG =  True
		
		self.num_inputs = self.config['num_inputs']		# number of input signals
		self.targets = 1							# number of reference signals
		
		#  Number of layers is signal + input + possible hidden + output 
		#						(min is 3: signal, input and output)
		self.layers = self.config['layers']
		self.num_layers = self.layers['count']
		if self.num_layers < 2 : 
			print ('Error: number of layers is less than 2')
			pygame.quit()
			sys.exit(0)
		self.learning_rate = self.config['learning_rate'] 	# somewhere around 0.001
		self.cycle_limit = self.config['cycle_limit']     	# maximum number of steps
		self.error_limit = self.config['error_limit']	   	# return once error is smaller than this value
		self.error = self.error_limit + 1
		
		# create the layers, # neurons = # inputs
		self.layers = list()
		for id in range(self.num_layers) :  # 0, 1, ...
			layer = self.structure['layers'][str(id)]
			if layer == 'error' :
				error_exit()
			self.layers.append( layer( layer) )
		self.sensor_inputs = list()

	# DEF: get_weights def of class NNBasic
	# Desc: Used to print weights or preserve weights in a file 
	# Return: a lsit of lists per layer of weights in each layer
	# Usage: print_state
	def get_weights( self ) :
		weights = list()
		for layer in self.num_layers :
			# get and append a list of weights for each layer
			weights.append (self.layers[layer].getWeights)
		return weights

	# DEF: get_outputs def of class NNBasic
	# Desc: Used to print outputs or to preserve outputs in a file
	# Return: an list of lists of outputs in each layer
	# Usage: print_state
	def get_outputs( self ) :
		outputs = list()
		for layer in self.num_layers :
			outputs.append (self.layers[layer].getOutputs)
		return outputs

	# DEF: print_error def of class NNBasic
	# Usage: print_state - can be used publically
	def print_error(self):
		if self.TEST :
			print('Error: {}'.format(self.error_return))

	# DEF: print_state def of class NNBasic
	# Desc: print current state for each layer/neuron - input/prev output, bias, weights 
	# 		then output value. This is called for each step if TEST is true.
	# Usage: public - for test and experimenting
	def print_state(self):
		if self.TEST == False : return
		
		outputs = self.getOutputs()
		weights = self.getWeights()	

		print('#### Current State ###')
		for layer in self.num_layers :
			print('Layer {} bias: {}'.format(layer, self.layers[layer].get_bias()))
			for neuron in self.layers[layer].num_neurons :
				if layer == 1 :
					for input in self.num_inputs :
						print('    Input {} = {}, weight = {}'.format(input, self.sensor_inputs[input], 
							weights[layer[input]]))
				elif layer < self.num_layers :
					print('Input {} value={} weight={}'.format(self.outputs[layer-1][self.outputs] ))
				else :
					print('Layer {} Output: {}'.format(layer, self.outputs[layer]))
		print('Expected Result: {}'.format(self.expected_results))
		self.print_error()

    # DEF: adapt def of class NNBasic
	# Desc: Input is a configuration dictionary, a set of values and a reference.
	# 		The objective is to adapt until the output matches the reference input
	# 		by looping steps until minimum error or max cycle count per the config.
	# 		At exit, record the input values, reference value and all weights.
	#		Config file parameter 'action' is either 
	#			"find" - start with random weights and adapt to the input values.
	#			"match" - load weights and reference from a saved file then
	#			 	change inputs until output matches the reference.
	# Parm: sensor_inputs - external values, adapt until the weighted output matches the reference.
	# Parm: reference value, the target value.
	# Parm: expected_result - output is almost 0 per goals in the config file.
	# Return: exits when output almost matches reference value. 
	#		Then all weights and the reference are saved to a JSON file 
	#       to be used to detect a set of input values.
	# Usage: called by class init as triggered by a user command.
	def adapt(self, sensor_inputs):
		self.sensor_inputs = sensor_inputs
		error_return = self.error_limit + 1
		self.target_input = sensor_inputs[0]
		cycle_count = 0
		while self.error > self.error_limit and cycle_count < self.cycle_limit :
			vehicle_command = self.step( sensor_inputs )
			cycle_count += 1
		return

    # DEF: step def of class NNBasic
	# Desc: one step in adaptation to new input data.
	#		Adapts weights to data, returns vehicle command
	# Assumptions: 
	#		AI structure is set by a config file.
	#		Output layer has num_outputs neurons where num_outputs
	#			is the number of training sets
	# Parm: inputs - list: state of the environment
	#		There is a set of inputs for each target output.
	#		note: sensor_inputs must be normaized by magnitude
	# Return: vehicle command, complex, of direction and speed 
	# Usage: public and adapt
	def step(self, inputs ):
		if self.TEST :
			self.stepped = True
		
		# Signal Se is the received exit signal. 
		# Signal Sc is the computed exit signal.
		# N is the computed sum of hill signals, the noise.
		# The output of the last neuron is Sc + N.
		#
		# Set the output goal to Se / (Sc + N), i.e. reference / output.
		# Ideally N is 0 so the target value is Se / Sc == 1 at the exit.
		# If not yet at the exit, the error is the speed and direction to move.
		# So the error at the output is 1 - output.
		# Use the error to compute inner layer weight changes.

		# feed forward: propagate inputs through all layers
		final_outputs = inputs
		for layer in self.layers : 
			final_outputs = layer.feed_forward(final_outputs)
 
		# now the inputs have been passed through all neurons and
		# the final output is available. So:
		# backward propagation: feed the error back through the layers
		# to adjust the weights
		output_error = calculate_total_error(self.expected_result)
		if output_error < self.error_limit :
			self.error_return = output_error
			return

		for layer in range[self.num_layers, -1, -1] : 
			input_errors = layer.feedback(input_errors)
		
			if layer == self.num_layers - 1: # then this is the output layer
				for n_idx in range(len(layer.neurons)):
					# ∂E/∂zⱼ - pd of output errors wrt output neuron total net input
					neuron = layer.neurons[n_idx]
					pd_errors_wrt_outputs[layer, n_idx] = \
						(neuron.output - targets[0]) * neuron.output * (1 - neuron.output)			

			else :
				for n_idx in range(len(layer.neurons)):
					# Calculate the derivative of the error with respect to 
					# the output of each neuron
					# dE/dyⱼ = Σ ∂E/∂zⱼ * ∂z/∂yⱼ = Σ ∂E/∂zⱼ * wᵢⱼ
					for o in range(len(layer.neurons)):
						# get error from layer + 1 to weight n_idx
						pd_errors_wrt_outputs[layer, n_idx] += \
							pd_errors_wrt_output_neuron_total_net_input[o] \
							* next_layer.neurons[o].weights[n_idx]
				# AND save this value for earlier layers
				# so each layer/neuron has an error feedback value
				# ∂E/∂zⱼ = dE/dyⱼ * ∂zⱼ/∂
				pd_errors_wrt_hidden_neuron_total_net_input[h] = \
					pd_error_wrt_hidden_neuron_output * \
					self.hidden_layer.neurons[h].calculate_pd_total_net_input_wrt_input()

		# After error is propagated and recorded back through all layers,
		# update the weights for each layer/neuron.
		output_layer = self.layers[ self.num_layers - 1 ]
		for neuron in output_layer.neurons :
			for w_index in range(len(neuron.weights)) :
                # ∂Eⱼ/∂wᵢⱼ = ∂E/∂zⱼ * ∂zⱼ/∂wᵢⱼ
				pd_error_wrt_weight = pd_err_wrt_input[neuron] * neuron.pd_input_wrt_weight(w_index)
				#  Δw = α * ∂Eⱼ/∂wᵢ change the output neuron weights:
				neuron.weight[w_index] -= self.LEARNING_RATE * pd_error_wrt_weight
		
		# now update neuron weights	
		pd_err_wrt_input[neuron]
		for layer in range[self.num_layers - 1, -1, -1] : # walk backwards through each layer
			layer.update_weights()

	# DEF: calculate_total_error def of class NNBasic
	# Desc: calculate error over all input sets. NOTE: This code
	# 		assumes that there is only one input set and one output.
	# Uses: self.expected result
	# Return: sum of errors for all input sets.
	# Usage:                                                                            
	def calculate_total_error(self):
		final_output = self.layers[self.num_layers - 1].neuron[0].output
		self.error = 0.5 * (self.expected_result - final_output) ** 2
		return self.error
			
# CLASS: Layer of module NNBasic
# Desc: a vertical layer of neurons. Create and support the neurons of this layer
# Usage: NeuralNet.init
class Layer:
	# DEF: __init__ def of class Layer
	# Parm: layer_id - the numeric id of this layer, a string
	# Parm: layer_dict - a dictionary describing the layer:
	#	{
	#		"type" : "input",
	#		"bias" : "0,0",
	#		"inputs" : // a dictionary
	#			"count" : 3, // number of inputs
	#			"0" :  // for inputs there are no neurons
	#			{  // but there are destinations into layer 1
	#			"destination count" : 3,
	#			"0" : // layer number
	#			{	// layer 1, first neuron, first  input
	#				"layer" : 1,
	#				"neuron" : 0,
	#				"input" : 0
	#			},
	# Usage: public
	def __init__(self, layer_id, layer_dict):
		self.bias = layer_dict['bias']
		self.layer_type = layer_dict['type']  # type is  sensor, input, hidden, or output
		self.neurons = list()   # neurons in this layer
		
		if self.layer_type == 'sensor' :
			# create the sensor layer (no neurons)
			inputs = layer_dict['inputs'] # input dictionary
			for input_id in range(inputs['count']) :
				destinations = str(input_id)
					
		elif self.layer_type == 'input':
			# create the input layer neurons
			neuron_dict = layer_dict['neurons']
			for id in range(neuron_dict['count']):
				# pass a neuron dictionary to create a new neuron
				self.neurons.append( Neuron(self.bias, neuron_dict[str(id)]) )
			
		elif self.layer_type == 'hidden':
			# create the hidden layer neurons
			neuron_dict = layer_dict['neurons']
			for id in range(neuron_dict['count']):
				# pass a neuron dictionary to create a new neuron
				self.neurons.append( Neuron(self.bias, neuron_dict[str(id)]) )
				
		elif self.layer_type == 'output' :
			# create the output layer neurons
			self.num_neurons = layer_dict['neurons']
			for id in range(neurons['count']):
				self.neurons.append(Neuron(self.bias, neuron_dict[str(id)]) )
				
		else :
			print('invalid layer type {}'.format(self.layer_type))
			return 'error'
		
    # DEF: feed_forward def of NNBasic subclass Layer
	# Parm: inputs - set of input values that go to each neuron in the layer
	# Return: a list of computed output values, one per neuron in this layer 
	# Usage: called by step 
	def feed_forward(self, inputs):
		for neuron in self.neurons:
			neuron.feed_forward

	# DEF: feed_back def of NNBasic subclass Layer
	# Desc: for each neuron sum the errors from connected inputs
	# 		in the next layer. Use the sum to modify the input weights.
	# Parm: next_layer_error_list - list of all errors at inputs of
	#		the next layer. 
	# Parm: layer_output_map - dictionary of how neuron outputs are
	#		connected to the next layer inputs so that the input error 
	#		values can be accessed.
	# Return - list of all error values by neuron:input
	def feed_back(self, next_layer_error_list ):
		error_list = list()
		for neuron in self.neurons:
			neuron.feed_back(self, self.layer_dict)
			for neuron in layer.neurons :
				# error = error in next neuron output * weight into next neuron
				for w_index in range(len(neuron.weights)) :
					# ∂Eⱼ/∂wᵢ = ∂E/∂zⱼ * ∂zⱼ/∂wᵢ
					# pd_error_wrt_weight = 
					#     pd_errors_wrt_hidden_neuron_total_net_input[h]
					#     * self.hidden_layer.neurons[h]
					#     .calculate_pd_total_net_input_wrt_weight(w_ih)
					pd_error_wrt_weight = pd_err_wrt_input[neuron] \
						* neuron.pd_input_wrt_weight(neuron.weights[w_index])
						
					# Δw = α * ∂Eⱼ/∂wᵢ
					neuron.weights[w_index] -= self.LEARNING_RATE * pd_error_wrt_weight
					# Compute error at the output of each neuron.
		# Return error in Neuron/input part of dictionary
		#
		# Adjust neuron input weights per the error at the output.
		# 
		# The error at the last neuron output is (target value - neuron output).
		# Each input weight is fully affected (sensitive to) this error.
		# But, the weights of the neurons in the previous layer are not as
		# effiective at changeing the output. The difference is the weight between
		# the two neurons which reduces the error that is fed back.
		#
		# Walk the error from output layer back to input layer and reduce the  error
		# by the input weights it passes through. Use the resulting error value to
		# reduce the input weights at each layer.
		#
		# If output goes to multiple neurons then use the sum of errors.
		# Return the errors in the input dictionary
		#
		# In each layer do one neuron at a time. For each neuron do one input at a time.
		#	The neuron feeds back the same error from the output. Then the input weight
		#	modifies that error and saves it for the neurons of the previous layers.
		#	
		#	previous layer - look forward to each output connection. Get the error (add them up if 
		#	multiple inputs connect back to one output. use it to record the errors for this layer.
		#	
		# Must go all the way through first to compute the errors. Then go back through to
		# modify the  weigths.
		return error_list

	# DEF: get_outputs def of NNBasic subclass Layer
	# Return: list of neuron outputs
	# Usage: public (print_state of class NeuralNet)
	def get_outputs(self):
		outputs = list()
			# append a list of weights for each neuron
		for neuron in self.neurons:
			# append output of each neuron to a list
			outputs.append( neuron.output )
		return outputs

	# DEF: get_weights def of NNBasic subclass Layer
	# Desc: return a dictionary describing all neurons
	# Return: dictionary of neurons
	# Usage: public (print_state of class NeuralNet)
	def get_weights(self) :
		weights = {}    # a dictionary
		for neuron in self.neurons:
			# append a list of complex weights for each neuron
			weights.append( neuron.get_weights())
		return weights

# CLASS: Neuron class of module NNBasic
# Desc: a single AI neuron - all weights and calculations
# Usage: Layer
class Neuron:
	# DEF: __init__ of class Neuron
	# Desc: create a neuron, connect outputs to next layer, set/adjust weights
	# Parm: layer_bias - a complex value
	# Parm: neuron_dict - dictionary of neuron configuration
	#		contains: 
	#		dictionary inputs:
	#			"count" : 3,
	#			"0" :  ["0,0.5"], // Complex weight initial value
	#
	#		dictionary destinations, each output to 
	#		multiple inputs in the next layer:
	#			"count" : 1, 
	#			"0" :
	#				{
	#				"neuron" : 0,
	#				"input" : 0
	#				}
	#			}
	# Parm: next_layer_error_list - list of reflected error at every input
	#		of the next layer.
	# Parm: target_output - value of final neuron output
	# Usage: created/set by class Layer
	def __init__(self, layer_bias, neuron_dict, layer_type, next_layer_error_list, target_output):
		self.bias = layer_bias		# complex
		self.input_dict = neuron_dict['inputs']
		self.type = layer_type		# "input", "hidden" or "output"
		self.next_layer_error_list = next_layer_error_list
		self.num_inputs = self.input_dict['count']
		self.error_list = [] * self.num_inputs    # feedback of errors
		self.inputs = []
		self.target_output = 1      # S/(S+N) where N == 0
		
		
		# sum of errors at next layer inputs back to this output
		self.dest_dict = neuron_dict['destinations']
		self.dest_count = self.dest_dict['count'] # number of neuron output destinations
		self.output = (0 + 0j)
		
		self.err_at_inputs = [(0 + 0j)] * self.num_inputs # one err for each neuron input
		self.weights = [(0 + 0j)] * self.num_inputs    # each input has a weight
	
	# DEF: feed_back def of class neuron
	# Desc: for each neuron sum the errors from connected inputs
	# 		in the next layer. Use the sum to modify the input weights.
	# Parm: next_layer_error_list - list of all errors at inputs of
	#		the next layer. 
	# Parm: layer_output_map - dictionary of how neuron outputs are
	#		connected to the next layer inputs so that the input error 
	#		values can be accessed.
	# Return - list of all error values by input
	def feed_back(self) :
		
		if self.type == 'output' :
			output_error = 0.5 * (self.target_output - self.output) ** 2
			for input_id in range(self.num_inputs):
				self.error_at_input[input_id] = error * self.weights[input_id]
				
				pd_errors_wrt_outputs[layer, n_idx] = \
					(neuron.output - targets[0]) * neuron.output * (1 - neuron.output)	
				for neuron in range(output_layer.neurons) :
					for w_index in range(len(neuron.weights)) :
						# ∂Eⱼ/∂wᵢⱼ = ∂E/∂zⱼ * ∂zⱼ/∂wᵢⱼ
						pd_error_wrt_weight = pd_err_wrt_input[neuron] * neuron.pd_input_wrt_weight(w_index)

		else: # feedback hidden layers		
			self.output_layer.neurons[o].weights[w_ho] -= self.LEARNING_RATE * pd_error_wrt_weight
			
			for input in range(self.inputs) :
				for dest_count in range( destinations['count']):
					dest_dict = destinations[str('dest_count')] 
					layer = dest_dict['layer']
					neuron = dest_dict['neuron']
					input = dest_dict['input']
					self.err_at_input[input] = error_at_output * self.weights[input]
						
						
		# Update input weights
		for inputs in range(len(self.inputs)):
			# ∂E/∂zⱼ - pd of output errors wrt output neuron total net input
			neuron = layer.neurons[neuron]
			pd_errors_wrt_outputs[layer, n_idx] = \
					(neuron.output - targets[0]) * neuron.output * (1 - neuron.output)			
			# save this value in the neuron, as input to next layer back
		return self.err_at_inputs

	# DEF: update_weights of class Neuron
	# After feedback errors are set through to next layer inputs, 
	#		update the weights for each neuron.
	# Parm: errorList - errors at each input of next layer
			
		""" ref
       # 3. Update output neuron weights
       for o in range(len(self.output_layer.neurons)):
           for w_ho in range(len(self.output_layer.neurons[o].weights)):

               # ∂Eⱼ/∂wᵢⱼ = ∂E/∂zⱼ * ∂zⱼ/∂wᵢⱼ
               pd_error_wrt_weight = pd_errors_wrt_output_neuron_total_net_input[o] * self.output_layer.neurons[o].calculate_pd_total_net_input_wrt_weight(w_ho)

               # Δw = α * ∂Eⱼ/∂wᵢ
               self.output_layer.neurons[o].weights[w_ho] -= self.LEARNING_RATE * pd_error_wrt_weight
		"""
		""" ref
		# 4. Update hidden neuron weights
        for h in range(len(self.hidden_layer.neurons)):
            for w_ih in range(len(self.hidden_layer.neurons[h].weights)):

                # ∂Eⱼ/∂wᵢ = ∂E/∂zⱼ * ∂zⱼ/∂wᵢ
                pd_error_wrt_weight = pd_errors_wrt_hidden_neuron_total_net_input[h] * self.hidden_layer.neurons[h].calculate_pd_total_net_input_wrt_weight(w_ih)

                # Δw = α * ∂Eⱼ/∂wᵢ
                self.hidden_layer.neurons[h].weights[w_ih] -= self.LEARNING_RATE * pd_error_wrt_weight		
		"""
		# ERROR?? following may need to be inserted in above code
        #      self.hidden_layer.neurons[h].weights[w_ih] -= self.LEARNING_RATE * pd_error_wrt_weight		

	# DEF: feed_forward of class Neuron
	# Desc: for an input set, compute the output as sum of weighted inputs.
	#     Always squash the output to limit range to 0:1. 
	#     Suggestion: (output - 0.5) * 2 gives range around -1 to 1.
	# Parm: inputs - either from sensors for from previous set of neurons
	# Return: output value 
	# Usage: self.feedforward
	def feed_forward(self, inputs):
		mag = self.bias
		for input in range(self.num_inputs):
			# if self.test : print("input {}, weight {}".format(inputs[input], self.weights[input]))
			mag += inputs[input] * self.weights[input]
			
		# squash the magnitude to 0..1 (NOT -1..1, angle does that)
		# This is the sigmoid function.
		return (1 / (1 + math.exp(-abs(mag))))

	# DEF: pd_error_wrt_input def of class Neuron
	# Desc: Partial Derivative of error with respect to total net input
	# 	Determine how much the neuron's total input has to change 
	#	to move closer to the target_output.
	# Parm: target_output - output value we want to get to 
	# Return: pd of error wrt total net input
	# Usage: NOT USED
	def pd_error_wrt_input(self, target_output): 
		pd_error_wrt_output = -(target_output - self.output)
		pd_total_net_input_wrt_input = self.output * (1 - self.output)
		return pd_error_wrt_output * pd_total_net_input_wrt_input;

# DEF: error_exit of module NNBasic.py
def error_exit() :
	pygame.quit()
	sys.exit(0)
	
#if 0
# ------------------Supplemental Information ---------------------------	
# ----------------------------------------------------------------------
# Following are other defs that have been integrated into the above code
# They are included here for your enlightenment via the excellent comments. 
# These are copied from Matt Mazur's open source code on github.
def calculate_output(self, inputs):
    self.inputs = inputs
    self.output = self.squash(self.calculate_total_net_input())
    return self.output

# Determine how much the neuron's total input has to change to move closer to the expected output
#
# Now that we have the partial derivative of the error with respect to the output (∂E/∂yⱼ) and
# the derivative of the output with respect to the total net input (dyⱼ/dzⱼ) we can calculate
# the partial derivative of the error with respect to the total net input.
# This value is also known as the delta (δ) [1]
# δ = ∂E/∂zⱼ = ∂E/∂yⱼ * dyⱼ/dzⱼ
def calculate_pd_error_wrt_total_net_input(self, target_output):
	return self.calculate_pd_error_wrt_output(target_output) * self.calculate_pd_total_net_input_wrt_input();
	# expands to:
	# -(target_output - self.output) * self.output * (1 - self.output)

# The partial derivate of the error with respect to actual output then is calculated by:
# = 2 * 0.5 * (target output - actual output) ^ (2 - 1) * -1
# = -(target output - actual output)
#
# The Wikipedia article on backpropagation [1] simplifies to the following, but most other learning material does not [2]
# = actual output - target output
#
# Alternative, you can use (target - output), but then need to add it during backpropagation [3]
#
# Note that the actual output of the output neuron is often written as yⱼ and target output as tⱼ so:
# = ∂E/∂yⱼ = -(tⱼ - yⱼ)
def calculate_pd_error_wrt_output(self, target_output):
	return -(target_output - self.output)

# The total net input into the neuron is squashed using logistic function to calculate the neuron's output:
# yⱼ = φ = 1 / (1 + e^(-zⱼ))
# Note that where ⱼ represents the output of the neurons in whatever layer we're looking at and ᵢ represents the layer below it
#
# The derivative (not partial derivative since there is only one variable) of the output then is:
# dyⱼ/dzⱼ = yⱼ * (1 - yⱼ)
def calculate_pd_total_net_input_wrt_input(self):
	return self.output * (1 - self.output)

#endif

