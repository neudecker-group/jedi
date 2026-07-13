from .jsonio import write_json as _write_json
from .jsonio import read_json as _read_json


# Make Jedi JSONable, similar to ASE's workaround


def write_json(self, fd):
    _write_json(fd, self)


@classmethod
def read_json(cls, fd):
    obj = _read_json(fd)

    if not isinstance(obj, cls):
        raise TypeError(...)

    return obj


def jsonable(name):

    def wrapper(cls):

        cls.ase_objtype = name

        if not hasattr(cls, "todict"):
            raise TypeError(...)

        cls.write = write_json
        cls.read = read_json

        return cls

    return wrapper
