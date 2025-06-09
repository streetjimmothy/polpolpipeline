import os
import git
from renderer import renderer
from scriptrunner import script_runner

class DisplayString():
	def __init__(self, value=""):
		self.value = value
	
	def append(self, value):
		self.value += ('\n' + value)

	def __str__(self):
		return self.value

class Menu:
	name = ""
	def __init__(self):
		self.display_string = DisplayString()

	def handle_input(self, user_input):
		raise NotImplementedError("Subclasses must implement this method")

	def init_run(self):
		self.display_string = DisplayString(f"{self.__class__.name} Menu")
		self.run()
		renderer.elements["menu"].content = str(self.display_string)

	def run(self):
		for key, value in self.options.items():
			self.display_string.append(f"{key}: {value}")

	def on_enter(self):
		pass
	
	def on_exit(self):
		pass

class MainMenu(Menu):
	name = "MAIN"

	def __init__(self):
		super().__init__()
		self.options = {
			"1": "Scripts",
			"2": "Test Menu",
			"4": "Exit"
		}

	def handle_input(self, user_input):
		if user_input == "1":
			return ScriptMenu.name
		elif user_input == "2":
			return TestMenu.name
		elif user_input == "4":
			return "QUIT"
		else:
			return MainMenu.name

class TestMenu(Menu):
	name = "TEST"
	def __init__(self):
		super().__init__()
		self.options = {
			"x": "Return",
		}
	
	def handle_input(self, user_input):
		if user_input == "x":
			print("Going back")
			return MainMenu.name
		elif user_input == "4":
			return "QUIT"
		else:
			return self.name

class ScriptMenu(Menu):
	name = "SCRIPTS"
	def __init__(self):
		super().__init__()
		self.options = {
			"1": "Show Scripts",
			"2": "Clone Scripts - This will replace any existing cloned scripts with clones from scripts/giturls.txt",
			"3": "Run Scripts",
			"r": "Return",
		}

	def handle_input(self, user_input):
		if user_input == "1":
			return ScriptListMenu.name
		elif user_input == "2":
			return ScriptDownloadMenu.name
		elif user_input == "3":
			return ScriptRunMenu.name
		elif user_input == "r":
			return MainMenu.name

class ScriptListMenu(Menu):
	name = "SCRIPTS_LIST"
	def __init__(self):
		super().__init__()
		self._display_string = DisplayString("Scripts Available:")
		for entry in os.scandir("./scripts"):
			if entry.is_file() and entry.name.endswith(".py"):
				self._display_string.append(f"\n{entry.name}")
			if entry.is_dir():
				self._display_string.append(f"\n{entry.name}/")
				for subentry in os.scandir(entry.path):
					if subentry.is_file() and subentry.name.endswith(".py"):
						self._display_string.append(f"\n\t{subentry.name}")
		self._display_string.append("Press any key to return to the previous menu")

	def handle_input(self, user_input):
		return ScriptMenu.name
	
	def run(self):
		self.display_string.append(str(self._display_string))

class ScriptDownloadMenu(Menu):
	name = "SCRIPTS_DOWNLOAD"

	def __init__(self):
		super().__init__()
		self.done = False

	def handle_input(self, user_input):
		return ScriptMenu.name

	def run(self):
		self.display_string.append("Downloading scripts...")
		if not self.done:
			with open("scripts/giturls.txt") as f:
				for line in f:
					repo = line.strip()
					self.display_string.append(f"\nCloning repository: {repo}")
					target_dir = repo.split("/")[-1].replace(".git", "")
					target_dir = f"scripts/{target_dir}"
					if os.path.exists(target_dir):
						self.display_string.append(
							f"\nDirectory {target_dir} already exists. Deleting...")
						try:
							os.rmdir(target_dir)
							self.display_string.append(f"\nDirectory {target_dir} deleted")
						except Exception as e:
							self.display_string.append(f"\nError deleting directory: {e}")
					try:
						git.Repo.clone_from(repo, target_dir)
						self.display_string += f"\nRepository cloned to {target_dir}"
					except Exception as e:
						self.display_string += f"\nError cloning repository: {e}"
				self.display_string += "\nDone"
				self.display_string += "\nPress any key to return to the previous menu"
				self.done = True
	
	def on_enter(self):
		self.done = False

class ScriptRunMenu(Menu):
	name = "SCRIPTS_RUN"

	def __init__(self):
		super().__init__()
		self.scripts = {}
		i = 0
		self.display_string = DisplayString("Scripts Available:")
		for entry in os.scandir("./scripts"):
			if entry.is_file() and entry.name.endswith(".py"):
				i+=1
				self.scripts[i] = entry
			if entry.is_dir():
				for subentry in os.scandir(entry.path):
					if subentry.is_file() and subentry.name.endswith(".py"):
						i+=1
						self.scripts[i] = subentry
		for key, value in self.scripts.items():
			self.display_string.append(f"\n{key}: {value.name}")

	def handle_input(self, user_input):
		if user_input in self.scripts:
			script_runner.run_script(self.scripts[user_input])
		return ScriptMenu.name

	def run(self):
		self.display_string.append(str(self._display_string))

class UI:
	def __init__(self):
		renderer.add_element("menu")
		self.current_screen = MainMenu.name
		self.screens = {}
		for menuclass in Menu.__subclasses__():
			self.screens[menuclass.name] = menuclass()

	def get_current_screen(self):
		return self.screens[self.current_screen]

	def get_current_screen_name(self):
		return self.current_screen

	def run_current_screen(self):
		self.screens[self.current_screen].init_run()
	
	def set_screen(self, screen_name):
		self.screens[self.current_screen].on_exit()
		self.current_screen = screen_name
		self.screens[self.current_screen].on_enter()

	def handle_input(self, user_input):
		print(f"User input: {user_input}")
		self.set_screen(self.screens[self.current_screen].handle_input(user_input))
