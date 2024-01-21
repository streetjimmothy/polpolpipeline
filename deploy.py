import os

parent_dir = os.path.join(os.getcwd(), '..')

for filename in os.listdir(parent_dir):
    print(filename)


parent_dir = os.environ.get('GITHUB_WORKSPACE')

for filename in os.listdir(parent_dir):
    print(filename)