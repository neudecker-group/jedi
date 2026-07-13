import json

from ase.io.jsonio import encode
from ase.io.jsonio import object_hook as ase_object_hook
from ase.utils import reader, writer


# Make Jedi JSONable, similar to ASE's workaround


def object_hook(dct):
    """Extension of ASE's object_hook supporting Jedi."""

    if dct.get("__ase_objtype__") == "jedi":
        from strainjedi.jedi import Jedi

        dct.pop("__ase_objtype__")
        return Jedi.fromdict(dct)

    return ase_object_hook(dct)


@writer
def write_json(fd, obj):
    fd.write(encode(obj))


@reader
def read_json(fd):
    return json.load(fd, object_hook=object_hook)
