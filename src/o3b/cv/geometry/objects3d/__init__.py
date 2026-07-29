import importlib
import pkgutil

discovered_plugins = {}
for finder, name, ispkg in pkgutil.iter_modules(__path__, __name__ + "."):
    if ispkg:
        try:
            discovered_plugins[name] = importlib.import_module(name)
        except Exception:
            pass
