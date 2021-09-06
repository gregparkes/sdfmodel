#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Methods relating to the caching using joblib library."""

import os
from typing import Callable
from joblib import load, dump


def cache(fn: str,
          f: Callable,
          debug: bool = True,
          *args,
          **kwargs):
    """Provides automatic caching for anything using joblib.

    Parameters
    ----------
    fn : str
        The name of the file to cache to, or read from. This is fixed. Include extension
    f : function
        A custom function returning the object to cache
    debug : bool
        Whether to print statements.
    *args : list, optional
        Arguments to pass to f(...)
    **kwargs : dict, optional
        Keyword Arguments to pass to f(...)

    Returns
    -------
    ca : cached element
        This can take many forms, either as list, tuple or dict usually
    """
    if os.path.isfile(fn):
        if debug:
            print("loading file '%s'" % fn)
        return load(fn)
    else:
        res = f(*args, **kwargs)
        if debug:
            # check that the folder path exists.
            print("writing file '%s'" % fn)
        dump(res, fn)
        return res


class Cacheable:
    """A cacheable object for holding file name and callable"""
    def __init__(self, fn, debug=True):
        self.fn = fn
        self.debug = debug

    def __call__(self, f, *args, **kwargs):
        return cache(self.fn, f, self.debug, *args, **kwargs)


class Reloader:
    """A simple __with__ class for cache contexts using a file name."""
    def __init__(self, fn, debug=True):
        self.fn = fn
        self.debug = debug

    def __enter__(self):
        return Cacheable(self.fn, self.debug)

    def __exit__(self, rtype, value, traceback):
        pass
