# Module: keyinput.py - handles all keyboard events
NOT USED - code moved to USER
# Description:
# 	Responds to all keyboard events. 
# 
# Classes:
# keyinput - 
#		maps keys to operations
#
# Program Structure:
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
#   History: Started 13 Jan 24 as a central switch
#   Author: Brad Denniston
#   Version: 0.1, 13 Jan 2024 
#	
# import python game module
#

import vehicle
import sensors
import neuralnet
import json
import cmath
import button
# ?? import pygame keep
import tools

class KeyInput: NOT USED - SEE USER
	def __init__(self, config, simmap) :
		self.DEBUG = config['debug']
	
	while True:
		# check for user action, take that action, then run a step
		#	else run a step.
		for event in pygame.event.get() :

        # is this event a mouse action?
        if event.type == pygame.MOUSEBUTTONUP:
            # reset the button image up to not active
            sim_map.menu.check_menu(event)
            continue

        elif event.type == pygame.MOUSEBUTTONDOWN:
		
            # it is a menu button, get button name 
            menu_return = sim_map.menu.check_menu(event)
			
            if menu_return != 'scene' :
                sim_map.mouse_button(event)
                if self.DEBUG print(menu_return)
            
            if menu_return == 'Run':
                running = True
                sim_map.menu_run()
				
            # load the saved file
            elif menu_return == 'Load':
                sim_map.menu_load()
            
			# save the current state 
            elif menu_return == 'Save':
                sim_map.menu_save()         
            
            elif menu_return == 'Manual':
                sim_map.menu_manual()

            elif menu_return == 'AI':
                sim_map.menu_ai()
            
            elif menu_return == 'Step':
                # if running then stop, then take a step
                if running :
                    running == False
                    sim_map.menu_stop()
                sim_map.menu_step()
    
            elif menu_return == 'Stop':
                running = False
                sim_map.menu_stop()

            elif menu_return == 'Help':
                running = False
                sim_map.menu_help()
                # show help screen till user enters 'return' or ESC
                # help_display = frame.help()
                
            elif menu_return == 'Exit':
                # do you want to save first?
                sim_map.menu_exit()
				pygame.quit()
				sys.exit(0)

        # no click on button, so check the direction keys
        elif event.type == KEYDOWN :
            sim_map.direction_key(event)
			
        elif event.type == QUIT:
            # system level as in ^C
            print('exiting')
            pygame.quit()
            sys.exit(0)
			
    # no valid user action, 
    # move the vehicle under AI or manual control
    if running :
        sim_map.menu_step()

    pygame.display.update()
    
    # slow everything down depending on size of fps
    fps_clock.tick(fps)
    # loop de loop
}
