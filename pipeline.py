if __name__ == '__main__':
	import os
	import logging
	import time
	import atexit

	from renderer import Renderer
	from input import Input

	from menu_classes import Menu, UI


	# logFormatter = logging.Formatter("%(asctime)s::%(levelname)s:%(message)s")
	# rootLogger = logging.getLogger()
	# rootLogger.setLevel(logging.DEBUG)

	# fileHandler = logging.FileHandler("pipeline.log")
	# fileHandler.setFormatter(logFormatter)
	# rootLogger.addHandler(fileHandler)


	# create scripts dir if it does not exist
	if not os.path.exists("scripts"):
		os.makedirs("scripts")

	def on_exit():
		#kill ui thread
		#should probably eventually do subprocess clean up here too
		print("Program is exiting...")

	# Register the on_exit function to be called upon program termination
	atexit.register(on_exit)

	renderer = Renderer()
	input = Input()
	ui = UI()

	while ui.get_current_screen_name() != "QUIT":
		if input.is_input():
			ui.handle_input(input.consume_input())
		ui.run_current_screen()
		time.sleep(0.1)