import os

for name, value in os.environ.items():
	print(f"{name}: {value}")