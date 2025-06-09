import sys
import threading
import multiprocessing
import time
import psutil
import humanize
from renderer import renderer

class Script():
	def __init__(self, name, script):
		self.name = name
		self.script_file = script
		
		self.last_poll_time = 0
		self.stats_element = renderer.add_element(self.name+"-stats", -1)

		self.output = []
		self.output_element = renderer.add_element(self.name+"-output")
		self.pipe, self.child_pipe = multiprocessing.Pipe()
		self.output_thread = threading.Thread(
			target=self.receive_output, 
			args=(self.pipe)
		)
		self.output_thread.daemon = True

	@staticmethod
	def _worker(script, shared_data, pipe):
		sys.stdout = pipe
		exec(open(script).read(), shared_data)
		pipe.close()
	
	def receive_output(self, pipe):
		while True:
			try:
				if pipe.poll():
					output = pipe.recv()
					self.output.append(output + "\n")
					if len(self.output) > 5:
						self.output.pop(0)
					self.output_element.content = "".join(self.output)
				if time.time() - self.last_poll_time > ScriptRunner.poll_interval:
					CPU_usage = self.psutil_process.cpu_percent(interval=1)
					memory_usage = self.psutil_process.memory_info().rss
					self.stats_element.content = (
						f"Name: {self.name}\t\t"
						f"CPU Usage: {CPU_usage}\t\t"
						f"Memory Usage: {humanize.naturalsize(memory_usage)}"
					)
					self.last_poll_time = time.time()
			except EOFError:
				break

	def run(self, shared_data):
		self.process = multiprocessing.Process(
			target=self._worker, args=(self.script_file, shared_data, self.child_pipe)
		)
		self.process.start()
		self.psutil_process = psutil.Process(self.process.pid)
		self.output_thread.start()

class ScriptRunner():
	_instance = None
	poll_interval = 0.1

	def __new__(cls, *args, **kwargs):
		if cls._instance is None:
			cls._instance = super(ScriptRunner, cls).__new__(cls, *args, **kwargs)
		return cls._instance

	def __init__(self):
		self.running_scripts = []
		self.shared_data = multiprocessing.Manager().dict()

	def run_script(self, script):
		script = Script(script.name, script)
		self.running_scripts.append(script)
		script.run(self.shared_data)
		

script_runner = ScriptRunner()