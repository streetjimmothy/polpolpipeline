import threading
import time
import os

class Element:
	def __init__(self, name):
		self.name = name
		self.content = ""
		self.priority = 0

class Renderer:
	_instance = None
	def __new__(cls, *args, **kwargs):
		if cls._instance is None:
			cls._instance = super(Renderer, cls).__new__(cls, *args, **kwargs)
			return cls._instance

	def __init__(self):
		self.elements = {}
		self.frame_time = 0
		display_thread = threading.Thread(target=self.render)
		display_thread.daemon = True
		display_thread.start()

	@staticmethod
	def clear_console():
		if os.name == 'nt':  # For Windows
			os.system('cls')
		else:  # For macOS and Linux
			os.system('clear')

	def add_element(self, name, priority=0):
		self.elements[name] = Element(name)
		return self.elements[name]

	SCREEN_REFRESH_RATE = 0.5
	def render(self):
		while True:
			if time.time() - self.frame_time > Renderer.SCREEN_REFRESH_RATE:
				Renderer.clear_console()
				for element in sorted(self.elements.values(), key=lambda element: element.priority):
					if '\033[2J\033[H' in element.content: #this is the ANSI escape sequence for clearing the console
						element.content = ""
					print(element.content)
				self.frame_time = time.time()


renderer = Renderer()
