# Class: User.py
# Desc: handle all user actions - keyboard and mouse
#
# Classes:
#	class User - used by NNBasic
#
# Permission is hereby granted, free of charge, to any person obtaning a copy
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
#	Author: Brad Denniston
#	Version: 0.1, 16 Jan 2024
#

# import python modules
# import python game module
import pygame
import json
# import NeuralNet module
import NNBasic

# CLASS: User
# Desc: Supports the user interface.
# Usage: handle all user inputs
class user:
    # DEF: User.__init__ 
	# Desc: This starts the routines that respond to user inputs.
	#   When one is detected we execute the task that responds to that key:
	#   	So this code is in a lazy loop as it keeps checking for a key hit. 
	#		At a key hit, everything else is ignored
	#   	until this new user direction is completed. Else there is a clock delay.
	#
	# Parm: config - dictionary of configuration data in JSON format loaded in main.
	# 	This tells the program what you want it to do. See the config file and 
	#	AiBasic documentation.
	#
	# Usage: NNBasic.py - the main process
	def __init__(self, config ):
		self.config = config
		self.DEBUG = config['debug']
		
		# start the neural net
		self.NeuralNet = NeuralNetBasic(self.config)

		# start the frame/second clock
		fps_clock = pygame.time.Clock()
		# how fast the simulation runs, default is 5
		fps = config[fps]
		# slow everything down depending on size of fps
		fps_clock.tick(fps)
		
		# Set up events
		# Turn off all events then allow ones we need
		pygame.event.set_allowed(None)
		allowed_events = (pygame.KEYDOWN, pygame.QUIT)
		for event in allowed_events:
			pygame.event.set_allowed(event)

		self.running = False

	# DEF:  KeyboardInput - all user keyboard events are processed here.
	#
	# Usage: user
	def KeyboardInput(self):
		running = False
		# now - we are set up. Get and process user events.
		while True:
			# check for user menu or keypad arrow action, then run a step
			event = pygame.event.get()
			if event.type == pygame.QUIT: # i.e. ^C
				print('exiting')
				exit_sim(0)
			keys = pygame.key.get_pressed()
			if keys[pygame.K_i]: # Input   - "i" prompts for input values and reference
				print("input an i?")
			if keys[pygame.K_x]: # Exit    - "x" or ^c to exit the simulation
				print('input is x, exiting')
				exit_sim(0)
			if keys[pygame.K_f]: # File    - "f filename", saves weights and reference value to a file
				print("input is f")
				break
				
			if keys[pygame.K_h]: # Halt    - "h" halt the AI, do not exit
				if self.running : 
					halt()
					self.running = False
				else:
					# shows list of commands
					ShowHelp()
				
			if keys[pygame.K_l]: # Load    - "l filename", load a saved file of weights and reference
				print('input is l')
			if keys[pygame.K_p]: # Plot    - "p", a graph screen plot of the output error while running
				print("input is p")
			if keys[pygame.K_r]: # Run     - "r" start or continue execution	
				print("input is r")
				self.running = True
				# if running stop, then take a step
				if self.DEBUG:
					print('Step')
				if running :
					running = False
					sim_map.menu_step()
					sim_map.neural_net_step()
			
			if keys[pygame.K_s]: # Step    - "s" stop if runninng, go one AI step when stopped
				print("input is s")
			if keys[pygame.K_t]: # Trace   - "t" show the output error at each step
				print("input is t")
			if keys[pygame.K_w]: # Weights - "w" show the current weights	
				print("input is r")	
			print("Input is not recognized")
		return
		
	# Def - ShowHelp
	#	Keyboard keys are:
	#		Input   - "i" prompts for input values and reference
	#		Exit    - "x" or ^c to exit the simulation
	#		File    - "f filename", saves weights and reference value to a file
	#		Halt    - "h" halt the AI, do not exit
	#		Help	- "h" when halted shows list of commands
	#		Load    - "l filename", load a saved file of weights and reference
	#		Plot    - "p", a graph screen plot the output error while running
	#		Run     - "r" start or continue execution	  
	#		Step    - "s" stop if runninng, go one AI step when stopped
	#		Trace   - "t" show the output error at each step
	#		Weights = "w" show the current weights
	# Usage - KeyboardInput of h when how running
	def ShowHelp() : 
		print("Input   - i - prompts for input values and reference")
		print("Exit    - x or ^c - to exit the simulation")
		print("File    - f filename - saves weights and reference value to a file")
		print("#Halt   - h - halt the AI, do not exit")
		print("#Help   - h - when halted shows list of commands")
		print("#Load   - l filename - load a saved file of weights and reference")
		print("Plot    - p - a graph screen plot the output error while running")
		print("Run     - r - start or continue execution")
		print("Step    - s - stop if running, go one AI step when stopped")
		print("Trace   - t - show the output error at each step")
		print("Weights = w - show the current weights")
		
	# DEF: exit def of class User
	# Desc: clean up before exit to system
	# Usage: user
	def ExitSim(self):
		pygame.quit()
		exit(0)	
                   
		

