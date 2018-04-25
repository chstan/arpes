import os

# note to Windows users: you need to specify your paths Windows style, this looks like:
# r'C:\Users\user\Documents\wherever'
# The r is for escaping the string, you can also do, if you like:
# 'C:\\Users\\user\\Documents\\wherever'
os.environ['ARPES_ROOT'] = r'/Users/chstansbury/PycharmProjects/python-arpes/' # <- change this out

DATA_PATH = None
# DATA_PATH = '/Users/chstansbury/Research/lanzara/data/' # <- change me too
