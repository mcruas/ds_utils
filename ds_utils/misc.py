import itertools
import yaml


def check_join(x,y) -> None:
    """Checks number of intersections and unique values. Prints on screen.

    Args:
        x,y (list, pd.Series, set): list of values   
    """
    xs = set(x)
    ys = set(y)
    print(f"X: {len(xs)} | Records {len(x)} ({len(xs) / len(x):.0%} unique)")
    print(f"Y: {len(ys)} | Records {len(y)} ({len(ys) / len(y):.0%} unique)")
    print(f"X \ Y: {len(xs)} -> {len(xs.difference(ys))} ({len(xs.difference(ys))/ len(xs):.0%})")
    print(f"Y \ X: {len(ys)} -> {len(ys.difference(xs))} ({len(ys.difference(xs))/ len(ys):.0%})")
    print(f"X intersection Y: {len(xs.intersection(ys))}")


def dict_product(dicts):
    """
    >>> list(dict_product(dict(number=[1,2], character='ab')))
    [{'character': 'a', 'number': 1},
     {'character': 'a', 'number': 2},
     {'character': 'b', 'number': 1},
     {'character': 'b', 'number': 2}]
    """
    return list((dict(zip(dicts, x)) for x in itertools.product(*dicts.values())))


# function to get current date and time
def get_current_time():
    from datetime import datetime
    return "[" + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "]"


def month_diff(x1, x2):
    return ((x1 - x2).dt.days / 30).round(0).astype(int)


def list_intersection(l1, l2):
    return list(set(l1).intersection(set(l2)))


def list_difference(l1, l2):
    return list(set(l1).difference(set(l2)))


def read_text(file_name, encoding='utf-8'):
    with open(file_name, 'r', encoding=encoding) as f:
        return f.read()


def read_yaml_file(file_name):
    with open(file_name) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    return data


def read_yaml(file):
    with open(file, 'r') as f:
        yaml_file = yaml.safe_load(f)
    return yaml_file


def read_config(config_file):
    config = read_yaml(config_file)
    exec = config.pop('exec', None)
    return config, exec
