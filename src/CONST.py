class _CONST:

    class ConstError(TypeError): pass

    def __setattr__(self, name, value):
        if name in self.__dict__:
            raise self.ConstError("ConstantError: Can't rebind const(%s)" % name)
        self.__dict__[name] = value


import sys

sys.modules[__name__] = _CONST()
