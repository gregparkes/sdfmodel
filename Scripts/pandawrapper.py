"""
A wrapping convenience class for Pandas.
"""
import re
import itertools as it
from functools import reduce
import numpy as np
from pandas import DataFrame, Series, Index, CategoricalDtype, read_csv, read_excel, read_sql
from pandas.core.dtypes.common import is_list_like
import glob
from typing import Callable, Tuple


def read(filename: str, *args, **kwargs):
    """Reads in one or multiple dataframes. accepts .csv, .xls, .xlsx, .sql"""
    glob_names = sorted(glob.glob(filename))
    if len(glob_names) == 0:
        raise IOError("No files selected with filename {}".format(filename))
    else:
        def _ext(s):
            return s.rsplit(".", 1)[-1].lower()
        pandas_types = {"csv", "xls", "xlsx", "sql"}

        def _fetch_db(fn):
            if _ext(fn) in pandas_types:
                return PandaWrapper.from_file(fn, *args, **kwargs)
            else:
                raise ValueError(f"file extension {ext} not recognized")

        results = list(map(_fetch_db, glob_names))
        if len(results) == 1:
            return results[0]
        else:
            return results


class PandaWrapper(DataFrame):
    """A super-class to wrap pandas.DataFrames with regex-style column selection and grouping."""

    def __init__(self, data=None, index=None, columns=None, dtype=None, copy=None):
        super().__init__(data, index, columns, dtype, copy)
        self._name = "df1"
        self._full_repr = False
        self._flex_regex = True
        self._view_as_union = True

    # properties

    @property
    def name_(self):
        """The name of the dataframe."""
        return self._name

    @name_.setter
    def name_(self, n: str):
        self._name = n if n is not None else "df1"

    @property
    def memory_(self):
        """Calculates the memory usage of the dataframe."""
        return "{:0.3f}MB".format(self.memory_usage(deep=False).sum() / 1000000.)

    @property
    def full_repr_(self):
        """Whether we fully represent the dataframe HTML output or not"""
        return self._full_repr

    @full_repr_.setter
    def full_repr_(self, fr: bool):
        self._full_repr = fr

    @property
    def unique_cols_(self):
        """Fetches the unique column names."""
        return list(self.columns.dropna().unique())

    # class methods

    @classmethod
    def from_file(cls, filename: str, name: str = None, *args, **kwargs):
        """Creates a PandaWrapper from file."""
        file_extensions_supp = {"csv": read_csv, "xls": read_excel,
                                "xlsx": read_excel, "sql": read_sql}

        directory, jname, ext = cls.__split_file_directory(filename)

        # read in dataframe.
        df = file_extensions_supp[ext](filename, *args, **kwargs)
        # now create PandaWrapper
        pr = cls(df)
        # override name using file
        pr._name = jname if name is None else name
        return pr

    @classmethod
    def wrap(cls, df: DataFrame, name: str = None):
        """Wraps the dataframe by calling the constructor"""
        my_new = cls(df)
        my_new.name_ = name
        return my_new

    @classmethod
    def __split_file_directory(cls, fn):
        fs = fn.replace("\\", "/").rsplit("/", 1)
        if len(fs) == 0:
            raise ValueError(f"filename '{fn}' not recognized")
        elif len(fs) == 1:
            directory = "."
            fname = fs[0]
        else:
            directory, fname = fs
        # just the name without the extension
        jname, ext = fname.split(".", 1)
        return directory, jname, ext

    @classmethod
    def __magic_type_string(cls):
        return {
            "int",
            "float",
            "bool",
            "object"
        }

    @classmethod
    def __magic_types_sel(cls):
        return {
            np.int,
            np.bool,
            np.float,
            np.float64,
            np.float32,
            np.float16,
            np.int64,
            np.int32,
            np.int16,
            np.int8,
            np.uint8,
            np.uint16,
            np.uint32,
            np.uint64,
            float,
            int,
            bool,
            object,
            CategoricalDtype
        }

    @classmethod
    def _set_like(cls, g):
        if isinstance(g, Index):
            return g.dropna().unique()
        else:
            return Index(sorted(set(g), key=g.index))

    @classmethod
    def _sorted_intersect(cls, *args):
        # intersection between two lists.
        if len(args) == 0:
            raise ValueError("no arguments passed")
        elif len(args) == 1:
            return PandaWrapper._set_like(args[0])
        else:
            return reduce(Index.intersection, map(PandaWrapper._set_like, args))

    @classmethod
    def _sorted_union(cls, *args):
        # intersection between two lists.
        if len(args) == 0:
            raise ValueError("no arguments passed")
        elif len(args) == 1:
            return PandaWrapper._set_like(args[0])
        else:
            return reduce(Index.union, map(PandaWrapper._set_like, args))

    @classmethod
    def _sorted_diff(cls, l1, l2):
        # difference between two lists.
        return sorted(set(l1) - set(l2), key=l1.index)

    @classmethod
    def _integrate_terms(cls, ta, tb):
        t1, op1 = ta
        t2, op2 = tb
        if op1 == '&':
            return (cls._sorted_intersect(t1, t2), op2)
        elif op1 == "|":
            return (cls._sorted_union(t1, t2), op2)
        else:
            return t1

    # hidden methods

    def _patcolumnmatch(self, pat: str, cols=None):
        _cols_list_like = cols if cols is not None else self.unique_cols_
        _patcomp = re.compile(pat)
        matches = sorted([n for n in _cols_list_like if re.search(_patcomp, n)],
                         key=_cols_list_like.index)
        if len(matches) > 0:
            return Index(matches, name=self.columns.name)
        else:
            return Index([], name=self.columns.name)

    def _extended_termmatch(self, pat: str, cols=None):
        _cols_list_like = cols if cols is not None else self.unique_cols_
        if pat.startswith("~"):
            _selection = self._patcolumnmatch(pat[1:])
            _diff = PandaWrapper._sorted_diff(_cols_list_like, _selection)
            return Index(_diff, name=self.columns.name)
        else:
            return self._patcolumnmatch(pat)

    # complex hidden functions

    def _view_str(self, pat: str):
        if self._flex_regex:
            terms = list(map(str.strip, re.split("[&|]", pat)))
            ops = re.findall("[&|]", pat)
            cols = self.unique_cols_
            if len(terms) == 0:
                return []
            elif len(terms) == 1:
                return self._extended_termmatch(pat, cols)
            else:
                matched_terms = [self._extended_termmatch(term, cols) for term in terms]
                # reduce operation on matched terms
                full = reduce(PandaWrapper._integrate_terms, it.zip_longest(matched_terms, ops))
                if isinstance(full, (list, tuple)):
                    return full[0]
                else:
                    return full
        else:
            return self._patcolumnmatch(pat)

    def _view_item(self, i):
        if i is None:
            return Index([], name=self.columns.name)
        elif is_list_like(i):
            return Index(i, name=self.columns.name)
        elif i in PandaWrapper.__magic_types_sel():
            return self.columns[self.dtypes.eq(i)]
        elif isinstance(i, str):
            if i in PandaWrapper.__magic_type_string():
                # then we have a type! convert this
                return self.columns[self.dtypes.eq(i)]
            elif i not in self.columns:
                return self._view_str(i)
            else:
                return Index([i], name=self.columns.name)
        else:
            raise ValueError(f"input pattern item '{i}' not recognized, choose a type or str, not type {type(i)}.")

    # public methods

    def view(self, pat):
        """Views the selected columns.

        Accepts 'extended regex' such as 'col1 & col2' for intersection-like operations
            spaces are ignored.

        Parameters
        ----------
        pat : type or string
            The pattern or name of column[s] to select

        Returns
        -------
        cols : Index
            The selected columns
        """
        return self._view_item(pat)

    def views(self, *pats):
        """Views the selected columns. Allows for multiple parameters

        Selections are joined together in union as A | B | C | ... | K

        Parameters
        ----------
        pats : args, type or string
            The pattern or name of column[s] to select

        Returns
        -------
        cols : Index
            The selected columns
        """
        if len(pats) == 0:
            raise ValueError("atleast one pattern must be given.")
        elif len(pats) == 1:
            return self._view_item(pats[0])
        else:
            match_groups = [self._view_item(i) for i in pats]
            # group together results
            if self._view_as_union:
                return self._sorted_union(*match_groups)
            else:
                return self._sorted_intersect(*match_groups)

    def ids(self):
        """Fetches the potential unique ID columns."""
        return self.columns[self.dtypes.eq(object)]

    def call(self, func_name: str, *args, **kwargs):
        """Calls a pandas.DataFrame function and wraps the result.

        Note that as this returns something, it does not modify in-place.

        Parameters
        ----------
        func_name : str
            The name of a PandaWrapper/DataFrame function
        *args, **kwargs : arguments to pass to said function

        Returns
        -------
        df : PandaWrapper copy result.

        Examples
        --------
        >>> new_wrapper = wrapper.call("copy")
        """
        if hasattr(self, func_name):
            result = getattr(self, func_name)(*args, **kwargs)
            return PandaWrapper.wrap(result, name=self.name_)
        else:
            raise ValueError(f"Function name '{func_name}' does not exist in pandas.DataFrame or PandaWrapper")

    def pipe_k(self, func: Callable, args_iter, **kwargs_fixed):
        """Calls .pipe k times, where k is the number of args_iter.

        Parameters
        ----------
        func : function(df, args_iter_i, **kwargs)
            A function taking a pandas.dataframe and returning a pandas.dataframe
        args_iter : list/tuple
            Arguments to iterate over for length k
        kwargs_fixed : dict of kwargs
            kwargs passed to every instance in pipe

        Returns
        -------
        df : a copy of dataframe

        Examples
        --------
        Mean of col1-col2 and col2-col3.
        >>> def aggregate_cols(df, selection):
        >>>     return df.assign(new_col=df[selection].mean(axis=1))
        >>> ndf = df.pipe_k(aggregate_cols, [['col1', 'col2'], ['col2', 'col3']])

        See Also
        --------
        pandas.DataFrame.pipe : piping custom functions.
        """
        result = self.copy()
        for arg in args_iter:
            result = result.pipe(func, arg, **kwargs_fixed)
        return PandaWrapper.wrap(result, name=self.name_)

    def rename_columns(self, *chain: Tuple[str, str]):
        """Renames columns using a chain of string replacement operations.

        Parameters
        ----------
        chain : list of 2-tuple
            chain contains a list of (<before>, <after>) str.replace operations. These are
            performed in-order across all column names.

        Returns
        -------
        df : PandaWrapper copy
        """
        if self.shape[1] > 1:
            _cols = self.columns
            s_new = list(reduce(
                lambda s_old, arg: map(lambda s: s.replace(*arg), s_old),
                [_cols, *chain]
            ))
            # make a copy
            new_wrapper = self.call("copy")
            # rename in-place.
            new_wrapper.call("rename", columns=dict(zip(_cols, s_new)), inplace=True)
            return new_wrapper
        else:
            raise ValueError("no columns to rename")

    # other internal overrides

    def __getitem__(self, i):
        # boolean series of indicators... i know, tiresome.
        if isinstance(i, Series) and i.dtype.kind == 'b':
            return PandaWrapper.wrap(super().__getitem__(i), name=self.name_)
        else:
            cols = self.view(i)
            if len(cols) <= 1:
                return super().__getitem__(cols).squeeze()
            else:
                return PandaWrapper.wrap(super().__getitem__(cols), name=self.name_)

    def __repr__(self):
        return f"PandaWrapper({self.name_}(n={self.shape[0]}, p={self.shape[1]}, mem={self.memory_}))"

    def _repr_html_(self):
        return self.__repr__() if not self._full_repr else super()._repr_html_()
