"""
SQL-style automatic merging.
"""

from __future__ import annotations
# itertools.
import itertools as it
import numpy as np
# reduce operation
from functools import reduce, partial
# typing for list of Dataframes
from typing import TYPE_CHECKING, List
# merge operation
from pandas.core.reshape.merge import merge as pmerge
# is_list_like checker
from pandas.core.dtypes.common import is_list_like
from pandas.core.dtypes.generic import ABCDataFrame

if TYPE_CHECKING:
    from pandas import DataFrame, Index, concat


def merge_auto(
    dfs: List[DataFrame],
    how: str = "inner",
    copy: bool = True,
    handle_duplicates: str = "end",
    verbose: int = 0,
) -> DataFrame:
    """
    Automatically merge multiple DataFrames with a database-style join.

    The join is done on columns or indexes. If joining columns on
    columns, the DataFrame indexes *will be ignored*. Otherwise if joining indexes
    on indexes or indexes on a column or columns, the index will be passed on.
    When performing a cross merge, no column specifications to merge on are
    allowed.

    Parameters
    ----------%s
    dfs : list-like of DataFrame
        DataFrame sets to merge together.
    how : {'left', 'outer', 'inner'}, default 'inner'
        Type of merge to be performed.

        * left: preserves keys in the first DataFrame, similar to a SQL left outer join;
          preserve key order.
        * outer: use union of keys across all frames, similar to a SQL full outer
          join; sort keys lexicographically.
        * inner: use intersection of keys from all frames, similar to a SQL inner
          join; preserve the order of the left keys.

          .. versionadded:: 1.2.0

    copy: bool, default True
        Whether to copy DataFrames at each iteration
    handle_duplicates: str, ['all', 'none', 'end'] default="end"
        'all': drops duplicates at each chain step, 'none' performs no duplicate removal, 'end' only drops
        'duplicates' at the end using the first label from the first dataset.
    verbose: int, default 0
        Prints out debug statements of increasing complexity up to 3.

    Returns
    -------
    DataFrame
        A DataFrame of the two or more merged objects.

    Raises
    ------
    ValueError
        If there is no intersection between the columns of a DataFrame pair set.

    See Also
    --------
    merge : Merge with a database-style join.
    merge_ordered : Merge with optional filling/interpolation.
    merge_asof : Merge on nearest keys.
    DataFrame.join : Similar method using indices.

    Examples
    --------
    """
    _dups_opts = {"all", 'none', "end"}

    if handle_duplicates not in _dups_opts:
        raise ValueError("`handle_duplicates` argument must be one of {}, not '{}'".format(_dups_opts, handle_duplicates))

    # check that dfs is a list, tuple or iterable
    if is_list_like(dfs):
        K = len(dfs)
        kws = dict(how=how, copy=copy, verbose=verbose,
                   allow_dups=False if handle_duplicates == 'all' else True)
        if K < 2:
            raise ValueError("dfs must contain at least 2 DataFrames")
        elif K == 2:
            result = _single_merge(dfs[0], dfs[1], **kws)
        else:
            # create a partial function passing kws.
            _merge_part = partial(_single_merge, **kws)
            result = reduce(_merge_part, dfs)
        return result
    else:
        raise ValueError(f"dfs must be list-like of DataFrames, not {type(dfs)}")


def _set_like(ind: Index) -> Index:
    """Computes the unique labels of the index."""
    return ind.dropna().unique()


def _has_majority_index_overlap(df1, df2) -> bool:
    """Checks whether the index objects overlap in a majority sense.

    We do this by comparing the shape size being
     greater than the half-average
    """
    if df1.index.dtype.kind in "OU" and df2.index.dtype.kind in "OU":
        ins = df1.index.intersection(df2.index)
        return ins.shape[0] > (np.nanmean((df1.shape[0], df2.shape[0])) // 2)
    else:
        return False


def _intersecting_pairs(df1, df2):
    """Given two dataframes, find all pairings between columns of interest."""
    df1_cols = df1.columns[df1.dtypes.eq(object)]
    df2_cols = df2.columns[df2.dtypes.eq(object)]
    """calculate pairings: we set this to a tuple rather than 
    iterable for later, although this ls less optimal for 
    performance. """
    _pairs = tuple(it.product(df1_cols, df2_cols))
    # create an empty numpy array to hold the pairing counts.
    _overlap = np.zeros((df1_cols.shape[0] * df2_cols.shape[0], 2))
    """Here we drop missing values and duplicates for each column
    rather than repeat this in the product iteration. This is a performance/memory
    trade-off that may not be suitable for large dataframes.
    """
    c1 = {x: df1[x].dropna().drop_duplicates() for x in df1_cols}
    c2 = {x: df2[x].dropna().drop_duplicates() for x in df2_cols}
    # iterate over the pairs, determine the non-duplicates
    for i, (p1, p2) in enumerate(_pairs):
        # fetch from pre-computed dictionary
        r1 = c1[p1]
        r2 = c2[p2]
        # compute the overlap using `isin` summing over boolean results.
        _overlap[i, 0] = r1.isin(r2).sum()
        _overlap[i, 1] = r2.isin(r1).sum()
    # return pairs tuple and mean between the 1 in 2 and 2 in 1 calculations
    return _pairs, _overlap.mean(axis=1)


def _maximum_pairs(df1, df2):
    """Discovers the maximum most-likely pairs between two dataframes"""
    _pair_names, _pair_count_av = _intersecting_pairs(df1, df2)
    # now, if we sum to 0, there is no crossover
    _num_shared_samples = np.sum(np.greater(_pair_count_av, 0))

    if _num_shared_samples == 0:
        raise ValueError(f"There is no crossover in column-data between df1 and df2 to merge upon")
    else:
        # pair names and pair count array should be the same length.
        _index_max = _pair_count_av.argmax()
        return _pair_names[_index_max], _pair_count_av[_index_max]


def _single_merge(df1: DataFrame,
                  df2: DataFrame,
                  how: str = "inner",
                  copy: bool = True,
                  allow_dups: bool = False,
                  verbose: int = 0) -> DataFrame:
    """Performing a single merge between two dataframes."""
    # check that both dfs are of type DataFrame.
    if not isinstance(df1, ABCDataFrame):
        raise ValueError(f"df1 parameter must be of type DataFrame, not {type(df1)}")
    if not isinstance(df2, ABCDataFrame):
        raise ValueError(f"df2 parameter must be of type DataFrame, not {type(df2)}")

    # if there is an overlap of indexes then we simply concat
    if _has_majority_index_overlap(df1, df2):
        # concat these together.
        df_new = concat((df1, df2), sort=False, join=how, axis=1, copy=copy)
        if verbose == 1:
            print("Merging df1 | df2 on indexes")
        elif verbose > 1:
            df1_name = df1.index.name if df1.index.name else "df1_index"
            df2_name = df2.index.name if df2.index.name else "df2_index"
            print(f"Merging df1 (`{df1_name}`) | df2 (`{df2_name}`) on indexes")
    else:
        # find the best pair of columns, _pair is a 2-tuple, _val is a float
        _pair, _val = _maximum_pairs(df1, df2)
        """Here we prepare a dictionary of arguments to pass to
        `pd.merge` depending on whether p1 and p2 are shared params."""

        merge_shared_params = (
            dict(on=_pair[0])
            if (_pair[0] == _pair[1])
            else dict(left_on=_pair[0], right_on=_pair[1]))
        merge_extras = dict(how=how, sort=False, left_index=False,
                            right_index=False, copy=copy)
        # drop duplicates within selecting columns prior to merging.
        if not allow_dups:
            _df1 = df1.drop_duplicates(subset=[_pair[0]])
            _df2 = df2.drop_duplicates(subset=[_pair[1]])
        else:
            _df1, _df2 = df1, df2
        # perform merge
        df_new = pmerge(_df1, _df2, **merge_extras, **merge_shared_params)
        # print statements
        if verbose > 0:
            df1_name = df1.name_ if hasattr(df1, "name_") else "df1"
            df2_name = df2.name_ if hasattr(df2, "name_") else "df2"
            if verbose == 1:
                print(f"Merging {df1_name} (`{_pair[0]}`) | {df2_name} (`{_pair[1]}`) on columns")
            elif verbose == 2:
                print("Merging {}[{},{}] (`{}`) | {}[{},{}] (`{}`) on columns".format(
                    df1_name, df1.shape[0], df1.shape[1], _pair[0],
                    df2_name, df2.shape[0], df2.shape[1], _pair[1]
                ))
            elif verbose > 2:
                print("Merging {}[{},{}] (`{}`) | {}[{},{}] (`{}`) -> {}/{:0.2f} on columns".format(
                    df1_name, df1.shape[0], df1.shape[1], _pair[0],
                    df2_name, df2.shape[0], df2.shape[1], _pair[1],
                    int(_val), _val / min(df1.shape[0], df2.shape[0])
                ))
    return df_new
