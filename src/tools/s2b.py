def str2bool(a):
    if isinstance(a, bool):return a
    if a.lower() in ('true', 'yes', '1', 't', 'y'): return True
    if a.lower() in ('false', 'no', '0', 'f', 'n'): return False
