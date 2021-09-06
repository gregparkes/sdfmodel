def control(import_set=None):
    if import_set is None:
        import_set = {"numpy", "scipy", "matplotlib"}

    import sys
    print("python: {}".format(sys.version))

    for module_name in import_set:
        # try to load in each module and complain if not
        try:
            module = __import__(module_name)
            if hasattr(module, "__version__"):
                print(f"{module_name}: {module.__version__}")
            else:
                print(f"{module_name} has no __version__ attribute.")
        except ImportError:
            raise ImportError("f{module_name} does not exist!")
