# Module: AI
# AIBasic.py - program start - there is no main()
#
# Description:
# AIBasic - provide a fundamental AI
#
# Class Structure:
# 	Pygame - user interface - also provides graphices for advanced releases
#   AIBasic - starts in this module. Calls NNBasic (the neural net).
# 		NOTE: there is no main. Execution starts here.
#		loads config which imports user definitions
#	User - waits for user keyboard inputs for next action
#	DEBUG - prints to screen per debug statements, activated in config
#   NNBasic - AI of neurons, handles predefined arithmetic pairs
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
#       Derived from AiLab
#   Author: Brad Denniston
#   Version: 0.1, 15 Jan 2024
#
# import python game module
import pygame
from pygame.locals import *

import sys
import time
import json
import user
import NNBasic  

pygame.init()

if not pygame.font.get_init() :
    pygame.font.init()
	
# Opening the config file per the name on the command line"
config_file = open(sys.argv[1])

# Read in user JSON file with definition of the AI structure and options
# To validate or print the config file use https://jsonformatter.org/ 
json_string = config_file.read()
config = json.loads(json_string)  # converts JSON to a dictionary

# check user events, set up and run the simulation as requested.
# user = user.UserInteraction( config )
user = user.user(config)
user.UserInteraction( )

# exit_sim def in main
# Desc: global, exits to OS 
def exit_sim():
    pygame.quit()
    exit(0)
