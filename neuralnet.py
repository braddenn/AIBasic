# Class: NeuralNet.py
#
# Classes:
#	class NeuralNet - Artificial Intelligence (AI) for input vectors, returns adjustment
#	class Layer - a vertical layer of neurons. Create and support the neurons of this layer.
#	class Neuron - a single AI neuron - all weights and calculations
#
# Desc: AI for AILab.
#	This is a general purpose AI where the structure 
#	(#layer, #neurons/layer) is defined in a JSON configuration file.
#   A layer consists of:
#       input sensors, neurons/weights, outputs, comparator, error computation, 
#       weights, biases, change results, and training inputs
#	For input sets (one of several input lists) compute an output set. 
#	There is output neurons that are trained for each input set.
#	Extension: applied to controlling vehicle movement
#	Extension: prints state after each step if self.DEBUG is True in config
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
# Comment references:
#	[1] Wikipedia article on Backpropagation
#		http://en.wikipedia.org/wiki/Backpropagation#Finding_the_derivative_of_the_error
#	[2] Neural Networks for Machine Learning course on Coursera by Geoffrey Hinton
#		https://class.coursera.org/neuralnets-2012-001/lecture/39
#	[3] The Back Propagation Algorithm
#		https://www4.rgu.ac.uk/files/chapter3%20-%20bp.pdf  
#
#	Author: Brad Denniston
#	Version: 0.4, 22 Nov 2018
#

import random
import math

# CLASS: NeuralNet
# Desc: Artificial Intelligence (AI) for input vectors, returns vehicle adjustment
# Usage: user
class NeuralNet:
	# DEF: __init__ def of class NeuralNet
	# Desc: The structure of this ANN is defined in a passed in dictionary
	#
	# Parm: config - dictionary of configuration data
	# Parm: sensors - provides sensor status
	# Parm: vehicle - the vehicle the NN is to move
	# Usage: user.py 
	def __init__(self, config, sensors, vehicle ):
		self.config = config
		self.DEBUG = config['debug']
		self.sensors = sensors
		self.vehicle = vehicle
		
		# These tags define constants in dictionary config
		self.cycle_limit = config['cycle limit']      	# maximum number of steps
		self.error_limit_mag = config['error limit mag']
		self.error_limit_angle = config['error limit angle']
		self.learning_rate = config['learning rate'] 	# somewhere around 0.001, directly affects weights
														# i.e. return once error is smaller than this value
		
		# These tags define constants in sub-dictionary self.target
		#	.type - if "exit" then out of the AI else internal destination
		#	.count - number of exits listed into internal dict
		# 	.output - real, imag, complex
		#	.destinations - a sub-sub dictionary of internal connections

		self.error_limit_mag = config['error limit mag']
		self.error_limit_angle = config['error limit angle']
		self.target = config['target']      # target is a dictionary
		
		#  Number of layers is input + possible hidden + output (min is 2, input and output)
		self.layers = config['layers']		# consists of a subdictionary for each layer
		self.test_set_count = config["test set count"]
		self.num_layers = self.layers['count']
		if self.num_layers < 2 : 
			print ('Adjust config: number of layers is less than 2. Input and output are required.')
			error_exit()
			
		self.num_outputs = self.test_set_count			# number of output signals
		self.target_output = 0
		self.error_return = 0
		
	# DEF: import_layers
	# Desc: Read in the user's descripton of the neural network, build the layers
	# Input is network layers: input, hidden(>=1), merge, compare, error
	# Usage: simmap
	def import_layers( self, config ):
		# create a list of layers from the config dictionary
		for id in range(self.num_layers) :
			layer_dict = self.layers[id]
			this_layer = Layer(id, layer_dict )
			if this_layer == 'error' :
				error_exit()
	
    # DEF: adapt def of class NeuralNet
	# Desc: The target is a signal, S, in the distance. A vehicle traverses
	# the field to get to the target, S. There are hills that block the vehicle.
	# They are detected and summed together to make Noise N.
	# Each adapt step responds to updated Signal and Noise data and returns a 
	# vehicle command to move (direction and speed).
	# An adapt step will consist of thousands of computations to walk along the
	# curve to the best solution: the return value set. The return set is fed 
	# back to the neurons to adjust the weights. Only the neuron weights
	# change until a minimum error is reached. Then the vehicle is moved.
    # 
	# 		Loop steps until minimum error or max cycle count
	# 		The goal is to avoid the hills by making S / (S + N) == 1

	# Parm: sensor_inputs - array of sensor inputs, target and hill noise.
	# Parm: target_input - pure target signal, S, and no hill noise, N. 
	# Return: vehicle_command - direction change and speed change.	
	# Usage: sim_map.menu_step
	def adapt(self, sensor_inputs, target_input, step):
		self.sensor_inputs = sensor_inputs
		# apply the sensor data to the input layer
		# this data never changes until the next adaptation
		if self.DEBUG:
			print('Adapting to sensor inputs:')
		endif
		for i in range(len(sensor_inputs)):
			# sens_mag, sens_ang = cmplxToRect(sensor_inputs[i])
			sens_mag = sensor_inputs[i]
			print( '{} : mag = {}'.format(i, sens_mag))
		targ_mag, targ_ang = cmplxToRect(target_input)
		if self.DEBUG:
			print( 'Target: mag = {} angle = {} deg'.format(targ_mag, targ_ang))
		
		# support user input of single step
		if( step == True ) :
			return self.step( sensor_inputs, target_input )
			
		self.errlim_mag = self.errlim_mag + 10
		self.errlim_ang = self.errlim_ang + 180
		cycle_count = 0
		self.error_return = complex(self.errlim_mag + 10, self.errlim_ang + 180)
		e_mag,e_ang = cmplxToRect(self.error_return)
		if self.DEBUG: print( 'e_mag:{}, e_ang:{}, cycle_count:{}'.format(e_mag, e_ang, cycle_count))
		if self.DEBUG: print( 'maglim:{}, anglim:{}'.format(self.errlim_mag, self.errlim_ang))

		# else cycle....
		while((e_mag > self.errlim_mag \
			or e_ang > self.errlim_ang) \
			and cycle_count < self.cycle_limit) :
			vehicle_command = self.step( sensor_inputs, target_input )
			cycle_count += 1
			if self.DEBUG:print( 'e_mag:{}, e_ang:{}, cycle_count:{}'.format(e_mag, e_ang, cycle_count))
		return vehicle_command
			
    # DEF: step - def of class NeuralNet
	# Desc: one step in adaptation to new input data.
	#		Adapts weights to data, returns vehicle command
	#		one step and adapt in loop till below error limit
	# Assumptions: 
	#		AI structure is set by a config file.
	#		Output layer has num_outputs neurons where num_outputs
	#			is the number of training sets
	# Parm: inputs - list: state of the environment (sensor data)
	#		There is a set of inputs for each target output.
	#		note: sensor_inputs must be normaized by magnitude
	# Parm: target_output - value of the target signal, no noise, normalized
	# Return: vehicle command, complex, of direction and speed 
	# Usage: menu_step
	def step(self, inputs, target_output ):
		
		# For signal Se as the independently received target signal. 
		# and Signal Sc as the computed exit/target signal.
		# and N is the computed sum of hill signals, i.e. the noise.
		# The output of the last neuron is Sc + N.
		#
		# Set the output goal to Se / (Sc + N), i.e. reference / output.
		# Ideally N is 0 so the target value is Se / Sc == 1 at the exit.
		# If not yet at the exit, the error is the speed and direction to move.
		# So the error at the output is 1 - output.
		# Use the error to compute inner layer weight changes.
		# feed forward: propagate inputs through all layers.
		# This applies the inputs to the  first layer outputs
		previous_outputs = inputs
		for layer in self.layers : 
			previous_outputs = layer.feed_forward(previous_outputs)
 
		# Now the inputs have been passed through all layers of neurons and
		# the final output is available.
		# So: on to backward propagation: feed the error back through the layers
		# to adjust the weights.
		output_error = self.calculate_total_error(target_output)	
		mag, angle = cmath.polar(output_error)
		if self.DEBUG:
			rmag = round(mag)
			angle = round(angle)
			if self.DEBUG:print('After feedforward, error is: {} at {} degrees'.format(rmag, angle))
			if self.DEBUG:print('errlim_mag is', self.errlim_mag)
		
		if mag < self.errlim_mag :
			self.error_return = output_error
			return -1

		# walk backward through the layers to adjust weights
		# First put output_error into final virtual layer, "error"
		self.layers[self.num_layers - 1].setFinalError(output_error)

		# Then start from there and walk backwards to update the weights for the next round.
        # Calculate the derivative of the error with respect to the output of each layer neuron
        # dE/dyⱼ = Σ ∂E/∂zⱼ * ∂z/∂yⱼ = Σ ∂E/∂zⱼ * wᵢⱼ
		# The error at the neuron output is passed backward 
		# and reduced by the input weight. It is stored in the neuron input dictionary.
		# Pass error back to input through rest of the layers

		for layer_idx in range(self.num_layers-2, 0, -1) : 
			self.layers[layer_idx].back_propagation()
			# so: start at output, put error into each neuron weight in layer/neuron dict
			# Then go back one layer and do same for each neuron
			
		return 0
		
	# DEF: calculate_total_error def of class NeuralNet
	# Desc: calculate error over all input sets. 
	#		NOTE: Currently, this code	assumes that there is 
	#		only one input set and one output.
	# Parm: target_output - goal of regression
	# Return: sum of errors for all input sets.
	# Usage: called by step                                                                           
	def calculate_total_error(self, target_output):
		# NOTE: this calc is for only one output neuron
		final_output = self.layers[self.num_layers - 2].neurons[0].output
		# add these up for all outputs
		return 0.5 * (target_output - final_output) ** 2

# CLASS: Layer of module NeuralNet
# Desc: a vertical layer of neurons. Create and support the neurons of this layer.
# Usage: NeuralNet.init
class Layer:
	# DEF: __init__ def of class Layer
	# Parm: layer_id - the numeric id of this layer, a string
	# Parm: layer bias - value
	# Parm: layer_dict - a dictionary describing the layer:
	# Parm: next_layer_dict - holds 
	# Parm: learning_rate - manages rate of weight change during feedback			
	# Usage: public - see *map.py
	def __init__(self, layer_id, bias, layer_dict, next_layer_dict, learning_rate):

		self.layer_id = layer_id
		self.neuron_bias = bias
		self.layer_dict = layer_dict
		self.next_layer_dict = next_layer_dict
		self.learning_rate = learning_rate
		self.layer_type = self.layer_dict['type']  # type is sensor, input, hidden, or output
		self.neurons = list()   # neurons in this layer
        # read the neurons for this layer
		self.neuron_dict = self.layer_dict['neurons']
		if self.DEBUG:print("266 " , self.neuron_dict)
		self.neuron_count = self.layer_dict['count']
		self.neuron_bias = bias 
		for id in range(self.neuron_count):
			self.neurons.append( Neuron( self.layer_id, self.neuron_bias, self.neuron_dict, \
				self.layer_type, self.learning_rate ))			
	
    # DEF: feed_forward def of class layer
	# Desc: called layer by layer, layer outputs are next layer inputs.
	# Parm: inputs - set of input values that go to each neuron in the layer
	# Return: a list of computed output values, one per neuron in this layer 
	# Usage: local
	def feed_forward(self, inputs):
		outputs = []
		# support user input of single step
		if self.DEBUG:print('281 layer {} Type {} Bias {}'.format(self.layer_id, self.layer_type, self.bias))
		for neuron_indx in range(self.neuron_count):
			outputs.append(self.neurons[neuron_indx].feed_forward( inputs, neuron_indx ))
		return outputs
	
	# DEF: setFinalError def of class layer
	# Parm: value of the final output error
	# Usage: local
	def setFinalError(self, output_error):
		error_neuron = self.neurons[self.neuron_count - 1]
		error_neuron.setFinalError( output_error )
	
	# DEF: back_propagation def of class layer
	# Desc: for each neuron, sum the errors from connected inputs
	# 		in the next layer. Use the sum to modify the input weights.
	def back_propagation(self):
		for neuron_id in range(self.neuron_count):
			neuron = self.neurons[neuron_id]
			neuron.back_propagation(self.neuron_dict[str(neuron_id)], self.next_layer_dict)
			
	# DEF: get_outputs def of class layer
	# Return: list of the output from each neuron 
	# Usage: public (print_state of class NeuralNet)
	def get_outputs(self):
		outputs = list()
		for neuron in self.neurons:
			# append output of each neuron to a list
			outputs.append( neuron.output )
		return outputs
				
# CLASS: Neuron of module NeuralNet
# Desc: a single AI neuron - all weights and calculations
class Neuron:
	# DEF: __init__ of class Neuron
	# Desc: create a neuron, connect outputs to next layer, set/adjust weights
	# Parm: layer_id - number of this layer
	# Parm: layer_bias - value
	# Parm: neuron_dict - dictionary of neuron configuration
	# Parm: layer_type: input, hidden, output
	# Parm: learning_rate - controls rate of weight change
	# Usage: created/referenced by class layer
	def __init__(self, layer_id, layer_bias, neuron_dict, layer_type, learning_rate):
		self.layer_id = layer_id
		self.bias = layer_bias
		self.neuron_dict = neuron_dict
		self.layer_type = layer_type
		self.learning_rate = learning_rate
		if self.DEBUG:print('302 - ', "layer id", layer_id, "bias", layer_bias, "neuron_dict", neuron_dict )
	
		if self.DEBUG:print("306 neuron_dict is:", neuron_dict)
		self.dest_dict = neuron_dict['destinations'] # list of output connections
		self.inputs_dict = neuron_dict['inputs']  # returns the backward prop err value
			
		self.type = layer_type
		# expected value at output layer, S/(S+N) where N == 0
		
		self.num_inputs = self.inputs_dict['count']
		self.inputs = [0j] * self.num_inputs
		
		# read in the initial weight values
		self.weights = [(0+0j)] * self.num_inputs    # each input has a weight
		for wt_cnt in range(self.num_inputs) :
			cmp_num_str = self.inputs_dict[str(wt_cnt)]
			self.weights[wt_cnt] = complex(cmp_num_str[0]) # convert string to complex
		self.output = (0+0j)
	
	# DEF: feed_forward of class Neuron
	# Desc: for an input set, compute the output as sum of weighted inputs.
	#     Always squash the output to limit the range to 0:1. 
	# Parm: inputs - either from sensors or outputs from previous layer
	# Return: inputs summed then squashed
	# Usage: layer.feedforward
	def feed_forward(self, inputs, neuron_index ):
		if self.type == "inputs" : # just return this one input signal
			self.output = inputs[0]
		else :
			self.output = self.bias
			for ndx in range(self.num_inputs):
				self.output += (inputs[ndx] * self.weights[ndx])
			# squash the magnitude to 0..1 (NOT -1..1, angle does that)
			# This is the sigmoid function.
			self.output = self.squash(self.output)
		
		# print each input, weight and the output
		if self.DEBUG:
			if self.DEBUG:print('Neuron {}'.format(neuron_index))
			for ndx in range(self.num_inputs):
				inmag,inang = cmplxToRect(inputs[ndx])
				wtmag,wtang = cmplxToRect(self.weights[ndx])
				print('    input {}: {} at {} deg  weight {} at {} deg.'.format(ndx, inmag, inang, wtmag, wtang))
			mag,ang = cmplxToRect(self.output)
			if self.DEBUG:print('    output {} at {} degrees'.format( mag, ang))
			if self.DEBUG:print()
		return self.output
		
	# DEF: squash def of class Neuron
    # Apply the logistic function to squash the output of the neutron
	# This is complex math so only squash the magnitude.
	# Usage: local
	def squash(self, total_net_input):
		mag, angle = cmath.polar(self.output)
		mag =  1 / (1 + math.exp(-abs(mag)))
		# back to complex with new magnitude
		return complex(mag, angle)

	# DEF: setFinalError def of class Neuron
	# Parm: value of the final output error
	# Usage: local
	def setFinalError(self, output_error):
		self.inputs_dict['0'][1] = output_error
		# this must be called for each layer/neuron. is it?
		if self.DEBUG:print('Output error: ', output_error)

	# DEF:  back_propagation of class neuron
	# Desc: gradient descent feed_back for each neuron output
	#		For each neuron the output error is the sum the errors 
	#		from next layer inputs that are connected to this neuron output.
	# 		New weight = input weight - sum * learning rate
	# 		To start, already put final error into the 'error' layer neuron 0, input 0.
	#		See setFinalError()
	# Parm: neuron_dict - provides destinations info for this neuron
	# Parm: next_layer_dict - dictionary of next layer neuron dictionaries 
	#		where the feedback error values are. 
	#		They were calculated during forward propagation.
	# Usage: class layer.back_propagation
	def back_propagation(self, neuron_dict, next_layer_dict) :
		dest_dict = neuron_dict['destinations']
		next_neurons_dict = next_layer_dict['neurons']

		# collect one sum of all destination inputs back to this neuron
		error_sum = (0+0j)
		
		# to collect and sum errors: walk destinations list in the config file
		# list entries are pairs: neuron id and input id
		#	"neurons" :
		#		"count" : 3,
		#		"0",
		#			{
		#			"destinations" :
		#				{
		#				"count" : 3, 
		#				"dests" : [0, 0, 1, 0, 2, 0]
		#				}
		
		# get the destinations list count (list size) and then the list
		dest_count = neuron_dict['destinations']['count']
		dest_dests_list = neuron_dict['destinations']['dests']
		
		# apply the feedback error at the output to modify each input weight
		for dests in range(0,dest_count,2):
			# list entries are pairs: neuron id and input id
			dest_neuron_id = str(dest_dests_list[dests])
			dest_input_id  = str(dest_dests_list[dests + 1])
			# "inputs" :
			#	{
			#	"count" : 1,
			#	"0" : ["1.0+0j","0+0j"] which are: input weight, error fed back to this input]
			#	},
			dest_neuron_inputs_dict = next_neurons_dict[dest_neuron_id]['inputs']
			error = complex(dest_neuron_inputs_dict[dest_neuron_id][1])
			# add all the next layer errors together to feed back total error for this neuron
			error_sum =  error_sum + error
			
		# Now adjust the input weights of this neuron
		if self.DEBUG:print('error sum, before weights:', error_sum, self.weights)
		for w_index in range(self.num_inputs):
			self.weights[w_index] -= error_sum * self.learning_rate
		if self.DEBUG:print('after weights:', self.weights)

# DEF: cmplxToRect
# Desc: convert complex # to mag,degrees with accuracy to 5 digits
# Parm: complex #
# Return: magnitude, angle
def cmplxToRect( cmplx ):
	mag, rads = cmath.polar(cmplx)
	degrees = round(angle(rads))
	mag = round(mag)
	return mag, degrees

# DEF: round
# Parm: value: small value to be truncated/rounded
# Return: value limited to 5 digits
def round(value):
	return( int(value * 10000) / 10000)	

# DEF: angle
# Desc: convert double radians to degrees in whole values  
# Parm: value - radians                                                                       
def angle(value):
	return int(10000 * (value * 180 / cmath.pi)) / 10000

# DEF: error_exit of module NerualNetCmplx.py
def error_exit() :
	pygame.quit()
	sys.exit(0)

# DEF: print_complex of module NeuralNet
# Desc: convert complex to mag,degrees and round to 5 places then print
# Parm: value - complex value
def print_complex(value):
	mag, radians = cmath.polar(value)
	angle = int(10000 * (radians * 180 / cmath.pi)) / 10000
	mag = int(10000 * mag) / 10000
	print(mag, angle)
		
# ------------------Supplemental Information ---------------------------	
#
# j - sets of inputs. 
# yⱼ - one output for each input set, j.
# Tⱼ - target output for an input set
# E - final error.
#
# ----------------------------------------------------------------------
# Backward Propagation: Determine how much a neuron's total input has 
# to change to move closer to the expected output 
# by feeding back the error, E, neuron by neuron. Each weight reduces
# the feed back error to the previous neuron. Then the weight is changed
# by the error amount.
#
# We have the partial derivative of the error, E, with respect 
# to the each output, yⱼ, for j input sets and j outputs:
#     (∂E/∂yⱼ)
# and the derivative of the outputs, yⱼ, with respect to the 
# total net input (dyⱼ/dzⱼ) 
#
# So we can calculate the partial derivative of the error with 
# respect to the total net input (the change). This value is also 
# known as the delta (δ):
# δ = ∂E/∂zⱼ = ∂E/∂yⱼ * ∂yⱼ/∂zⱼ
#

#def calculate_pd_error_wrt_total_net_input(self, target_output):
#    self.calculate_pd_error_wrt_output(target_output) \
#    * self.calculate_pd_total_net_input_wrt_input();
#
# which reduces to:
#    -(T - E) * E * (1 - E)
#
# The partial derivative of the error with respect to actual output
# then is calculated by:
#     2 * 0.5 * (T - E) ^ (2 - 1) * -1
#     = -(T - E)
#     = (E - T)
#
# Alternative, you can use (T - yⱼ), but then need to add it during backpropagation [3]
#
# Note that the actual output of the output neuron is often written as yⱼ and target output as tⱼ so:
# = ∂E/∂yⱼ = -(tⱼ - yⱼ)
#
#def calculate_pd_error_wrt_output(self, target_output):
#	return -(target_output - self.output)
#
# The total net input into a neuron is squashed using
# the logistic function to calculate the neuron's output:
# yⱼ = φ = 1 / (1 + e^(-zⱼ))
# Note that where ⱼ represents the output of the neurons in 
# whatever layer we're looking at and ᵢ represents the layer after it.
# For complex values ignore the phase, only use the magnitude. This is
# because squashing the magnitude prevents runaway feedback, squashing
# the phase does not.
#
# The derivative (not partial derivative since there is only one variable) of the output then is:
# dyⱼ/dzⱼ = yⱼ * (1 - yⱼ)
#
#def calculate_pd_total_net_input_wrt_input(self, output):
#	return output * (1 - output)


