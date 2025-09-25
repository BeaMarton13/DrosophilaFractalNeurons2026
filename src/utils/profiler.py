from pyinstrument import Profiler
import functools

def profile(func):
    """A decorator that profiles a function using pyinstrument."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        profiler = Profiler()
        profiler.start()
        
        result = func(*args, **kwargs)
        
        profiler.stop()
        
        print(f"\nProfiling report for function: '{func.__name__}'")
        print("--------------------------------------------------")
        print(profiler.output_text(unicode=True, color=True))
        
        return result
    return wrapper