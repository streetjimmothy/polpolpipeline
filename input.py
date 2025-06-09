import keyboard
import threading

class Input:
	def __init__(self):
		self.input_buffer = ""
		input_thread = threading.Thread(target=self.wait_for_input)
		input_thread.daemon = True
		input_thread.start()
	
	def peek_input(self):
		return self.input_buffer
	
	def is_input(self):
		return len(self.input_buffer) > 0

	def consume_input(self):
		input = self.input_buffer
		self.input_buffer = ""
		return input
	
	def wait_for_input(self):
		while True:
			self.input_buffer = keyboard.read_hotkey(False) #TODO: Input is passed to other applications (including the shell?) - change to True to fix
