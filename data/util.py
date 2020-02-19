'''
Utility functionalities.
'''
import time

class TimedList:
    '''List with added timestamps for each entry.'''
    def __init__(self, name=None, debug=True, values=None, times=None):
        self.name = name
        self.debug = debug
        self.values = list() if values is None else values
        self.times = list() if times is None else times
        if (values is not None and times is None) or (values is None and times is not None):
            raise Exception("Both values and times must be given.")
        if values is not None and times is not None:
            if len(values) != len(times):
                raise Exception("values and times must be of equal length.")
    
    def append(self, e):
        now = time.time()
        if self.debug:
            print(self.name, "=\t", e, "\tat time", now)
        self.values.append(e)
        self.times.append(now)