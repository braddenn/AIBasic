# class write.py 
# Desc: supports writing text to a popup
# Class: includes
#    Write - with logging or print support. Default is to print.
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
# Author: Brad Denniston
# Version: 0.3 - 13 jan 2020
# ------------------------------
# Class Help
#		used by 
# def display
# def draw
# def getSurfaces
class Help:
	#! ---------------------------
	#! DEF: __init__ for class Help
	def __init__(self, config):
		self.config = config
		self.height = 600  # gets changed in the program depending on space taken up by help
		self.surfaces = self.get_surfaces()
		self.font = pygame.font.SysFont("Arial",8) 
		self.text_size  = 20
		self.title_size = 30
		self.indext_size = 40
		self.slider_width = 10
		self.slider_gap_size = 5
		self.slider_length = 100
		self.width = 1000 + slider_width + slider_gap_size
		self.scroll_amount = 50
		self.color = { 
			'background': (120,120,120),
			'text' : (0,0,0),
			'slider' : (0,244,100),
			}

	# -------------------------------------------
	# DEF: display of class Help
	# Desc: Displays the help page on the given screen
	# Param: screen
	def display(self, screen):
        
		print ('Help selected')
		pygame.display.set_caption("DuneBuggy Help")
		pygame.display.set_mode((self.width, self.surfaces[0].get_height()))
		screen.fill(self.color['background'])
		self.height = screen.get_height()
		slider_range = (self.slider_gap_size + self.slider_length // 2,
						self.height - self.slider_gap_size - self.slider_length // 2)
		slider_center = slider_range[0]
		help_rect = self.surfaces[0].get_rect()  # initialises the help surface to be written
		help_rect.topleft = (self.help_gap_size, self.help_gap_size)
		screen.blit(self.surfaces[0], help_rect)  # puts help surface onto the screen
		self.draw(screen, self.surfaces[1], slider_center, slider_range)
		slider_last_turn = False/self.config.help_scroll_amount
		while True:
			events = pygame.event.get()
			if check_quit(events):
				break
			x, y = pygame.mouse.get_pos()
			if pygame.mouse.get_pressed()[0]:
				if slider_last_turn:
					y = max(min(y + slider_center - mouse_start, slider_range[1]), slider_range[0])
					self.draw(screen, self.surfaces[1], y, slider_range)
				elif -2 * self.config.slider_gap_size - self.config.slider_width < x - self.width < 0:
					slider_last_turn = True
					mouse_start = y
					if not slider_center - self.config.slider_length / 2 <\
						y < slider_center + self.config.slider_length / 2:  # if mouse was not clicked
						slider_center = y  # directly on top of the slider
			elif slider_last_turn:
				slider_last_turn = False
				slider_center += y - mouse_start  # reset the position of the slider
			if x > (self.width - self.config.help_slider_width - self.section_gap_size) / 2 - self.slider_gap_size:
				draw = False
				for e in events:
					if e.type == pygame.MOUSEBUTTONDOWN:
						if e.button == 4:  # if scrolled down
							slidercenter = max(slidercenter - self.ScrollAmount, sliderRange[0])
							draw = True
						if e.button == 5:  # if scrolled up
							slidercenter = min(slidercenter + self.ScrollAmount, sliderRange[1])
							draw = True
				if draw:
					self.draw(screen, self.Surfaces[1], slidercenter, sliderRange)
			pygame.display.update()
			fps_limiter.tick(self.config.FPS)
    
    # -------------------------------------------
    # DEF: draw of class Help
    # Desc: Draws the right hand side of text & slider at given levels
    # Param: screen - the frame to write to
    # Param: help_surface
    # Param: slider_center
	def draw(self, screen, help_surface, slider_center, slider_range):
        
		pygame.draw.rect(screen, self.config.help_color["Background"],  # draws over changing part of screen
                         ((self.width - self.config.slider_width - self.config.slider_gap_size)
                          // 2 - self.config.slider_gap_size, 0, self.width, self.height))
		pygame.draw.rect(screen, self.color["Slider"],  # draws slider
                         (self.config.slider_width - self.config.slider_gap_size - self.config.slider_width,
                          slider_center - self.slider_length // 2,
                          self.config.slider_width, self.config.slider_length))
		help_rect = help_surface.get_rect()
		text_range = (self.config.help_gap_size, help_surface.get_height()
                      - self.height + 2 * self.config.help_gap_size)
		top_y = text_range[0] - (text_range[1] - text_range[0]) * (slider_center - slider_range[0])\
                                // (slider_range[1] - slider_range[0])  # where the help surface is
		help_rect.topleft = (int((self.config.help_width - self.config.slider_width) // 2) + self.config.slider_gap_size, top_y)
		screen.blit(help_surface, help_rect)# sets position of help surface in relation to the screen
		pygame.display.update()

    # -------------------------------------------
    # DEF: get_surfaces of class Help
    # Desc: Gets the surfaces for the help screen. this needs to only be called once,
    #     and the surfaces saved to a variable, as it takes a while to run
    # Returns: array of help surfaces to be displayed via slider
	def get_surfaces(self):
		text = open("help.txt").read().split("++")  # split into the two sections
		for section in range(len(text)):
			text[section] = text[section].split("\n")  # splits into lines
		help_surfaces = []
        
		for section in text:
			extra = 0  # first time is to see how big the surface must be to fit the text,
			for _ in range(2):  # the second time it writes it onto a surface of that size
				help_surface = pygame.Surface(((self.width - self.config.slider_width)
                                               // 2 - self.config.help_gap_size - self.config.slider_gap_size,
                                               extra))
				help_surface.fill(self.config.help_color['Background'])
				extra = 0
				for line in section:
					if line.startswith("**"):  # bold text - titles etc.
						size = self.config.help_title_size
						line = line[2:]
					else:
						size = self.config.help_text_size
					indent = 0
					while line.startswith("--"):  # indented text
						indent += 1
						line = line[2:]
    	               # screen, x, y, text, color, size, max_len = None, gap = 0, 
    	               # font= self.config.font, rotate = 0, alignment = ("left", "top")):
					maxLen = (help_surface.get_width() - indent * self.config.help_indent_size) + \
						self.config.help_gap_size
					extra += self.write(help_surface, indent * self.config.help_indent_size, extra, line,
							self.color["Text"], size, max_len = maxLen )
				help_surfaces.append(help_surface)
		return help_surfaces