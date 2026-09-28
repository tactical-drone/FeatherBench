"""
Registry: a named slot where interchangeable parts live.

Every compartment (datasets, activations, optimizers, ...) is one Registry.
Adding a part is one decorator; the UI and the headless runner list whatever
is registered, so new parts show up everywhere automatically.

    from nncore import ACTIVATIONS

    @ACTIVATIONS.register("swish2")
    def swish2():
        return MySwish2()
"""


class Registry(dict):
    def __init__(self, kind, signature=""):
        super().__init__()
        self.kind = kind
        self.signature = signature  # human-readable contract, shown in errors

    def register(self, name, obj=None):
        """Use as @REG.register("name") or REG.register("name", obj)."""
        if obj is not None:
            self[name] = obj
            return obj

        def deco(fn):
            self[name] = fn
            return fn
        return deco

    def get(self, name):
        try:
            return self[name]
        except KeyError:
            raise ValueError(f"unknown {self.kind} '{name}' (have: {', '.join(self)})") from None

    def names(self):
        return list(self)

    def __repr__(self):
        return f"Registry({self.kind}: {', '.join(self)})"
