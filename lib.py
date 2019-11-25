# Commonly used functions in the project
from time import gmtime, strftime

def stringify(something):
    if type(something) == list:
        return [stringify(x) for x in something]
    elif type(something) == tuple:
        return tuple(stringify(list(something)))
    else:
        return str(something)
    
def log(*msg):
    msg = stringify(msg)
    print(strftime("[%H:%M:%S]", gmtime()), " ".join(msg))
    
log("Library functions loaded.")