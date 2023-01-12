# define seed
def set_seeds(libraries, seed=42):
    for li in libraries:
        eval(f"{li}.seed(seed)")

def dict2tuple(x: dict) -> tuple:
    """Converts a dictionary into an equivalent tuple structure

    Parameters
    ----------
    x : dict
        input dictionary

    Returns
    -------
    tuple
        Output tuple

    Examples
    --------
    FIXME: Add docs.

    """

    y = []
    for k, v in x.items():
        z = []
        z.append(k)
        if isinstance(v, dict):
            z.append(dict2tuple(v))
        else:
            z.append(v)
        y.append(tuple(z))
    return tuple(y)
