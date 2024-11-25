import time
from functools import wraps

def timer(enable = True, logger = None):
    def wrap(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if enable:
                start = time.time()
                if logger:
                    logger.info(f"Starting {func.__name__}")
                else:
                    print(f"Starting {func.__name__}")

                result = func(*args, **kwargs)
                end = time.time()
                if logger:
                    logger.info(f"{func.__name__} took {end - start} seconds")
                else:
                    print(f"{func.__name__} took {end - start} seconds")
                return result
            else:
                return func(*args, **kwargs)
        return wrapper
    return wrap