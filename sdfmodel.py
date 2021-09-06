"""SDFModel class wrapper for pre-trained models."""

import re
import warnings
import numpy as np
import pandas as pd
import joblib

from pandawrapper import PandaWrapper, read
from merge_auto import merge_auto


class SDFModel:
    """SDF Model wrapper class."""

    def __init__(self,
                 model="Models/xgb_basic.pkl",
                 hgnc_data="Data/hgnc_labels.csv",
                 sdf_data="Data/sPCA_dg_mca.csv"):
        """Instantiates a sequence-derived feature wrapper class.

        Parameters
        ----------
        model : str, xgb.XGBRegressor
            A pre-trained model
        hgnc_data : str, pd.DataFrame
            HGNC label dataset
        sdf_data : str, pd.DataFrame
            SDF dataset
        """
        # handle model
        if type(model) is str:
            self._model_fp = model
            self._model = None
        elif hasattr(model, "fit"):
            self._model = model
        elif model is None:
            self._model = None
        else:
            raise ValueError("model input not allowed")

        # handle hgnc
        if type(hgnc_data) is str:
            self._hgnc_fp = hgnc_data
            self._hgnc = None
        elif type(hgnc_data) is pd.DataFrame:
            self._hgnc = hgnc_data
            self._process_unique_labels()
        else:
            raise ValueError("hgnc_data input not allowed")

        # handle X
        if type(sdf_data) is str:
            self._X_fp = sdf_data
            self._X = None
        elif type(sdf_data) is pd.DataFrame:
            self._X = sdf_data
        else:
            raise ValueError("sdf_data input not allowed")

    @property
    def SDF(self):
        """Sequence-derived features"""
        return self._X

    @property
    def model(self):
        """Returns the saved model."""
        return self._model

    @property
    def hgnc(self):
        """HGNC labels"""
        return self._hgnc

    def load(self):
        """Load in datasets from file."""
        if self._X is None:
            self._X = read(self._X_fp)
        if self._hgnc is None:
            self._hgnc = read(self._hgnc_fp, low_memory=False)
            self._process_unique_labels()
        if self._model is None:
            self._model = joblib.load(self._model_fp)
        return self

    def is_loaded(self):
        """Whether the key variables are loaded and ready to do predictions."""
        return self._X is not None and self._hgnc is not None and self._model is not None

    def _process_unique_labels(self):
        # make dictionary dropped subsets for every column.
        self._hgnc_fast = {x: self._hgnc[x].dropna().drop_duplicates() for x in self._hgnc.columns}

    def _find_column(self, X):
        # find if labels belong to any column in HGNC
        nvals = np.zeros(self.hgnc.shape[1], dtype=np.int)
        for i, col in enumerate(self.hgnc):
            nvals[i] = self._hgnc_fast[col].isin(X).sum()
        # if any nval > 0, then we choose the max
        if np.any(nvals):
            col_id = np.argmax(nvals)
            return col_id, self.hgnc.columns[col_id]
        else:
            raise ValueError(f"no crossover between {X} and hgnc dataset")

    def _find_column_regex(self, X):
        # select 'object' only columns
        obj_cols = self.hgnc.columns[self.hgnc.dtypes == object]
        # find if labels belong to any column in HGNC
        nvals = np.zeros(obj_cols.shape[0], dtype=np.int)
        for i, col in enumerate(obj_cols):
            nvals[i] = self._hgnc_fast[col].str.strip().str.contains(X).sum()
        # if any nval > 0, then we choose the max
        if np.any(nvals):
            col_id = np.argmax(nvals)
            col_name = obj_cols[col_id]
            nX = self._hgnc_fast[col_name][self._hgnc_fast[col_name].str.strip().str.contains(X).fillna(False)]
            return col_id, col_name, nX
        else:
            raise ValueError(f"no crossover between {X} and hgnc dataset")

    def _calculate_seq_info(self, X, use_regex=False):
        # if not integrated handle this
        if not hasattr(self, "_intgr"):
            self._intgr = merge_auto([self.hgnc, self.SDF])
        # get the best column if available.
        if use_regex:
            # replace string X with a suitable list
            colid, col_name, X = self._find_column_regex(X)
        else:
            colid, col_name = self._find_column(X)
        # eliminate duplicates then select rows using this col name
        _intgr_sub = self._intgr.drop_duplicates(subset=[col_name])
        _select_bool = _intgr_sub[col_name].isin(X)
        _tot = _select_bool.sum()
        N = len(X)
        if _tot <= 0:
            raise ValueError("No samples remaining after duplicates dropped.")
        if _tot < N:
            warnings.warn(f"merging operation returns {_tot} out of {N} selections.", UserWarning)
        # return the selection information
        return _intgr_sub, _select_bool, col_name

    def get_X(self, X, regex=False):
        """Fetches sequence-derived feature information corresponding to given labels.

        Parameters
        ----------
        X : str or array-like of str (n,)
            Given HGNC, Uniprot, Refseq or Ensembl labels. Must match with something
            in SDFModel.hgnc DataFrame.
        regex : bool, optional, default=False
            If True, X must be a str and accepts regular expression.

        Returns
        -------
        df : pandas.DataFrame or PandaWrapper
            data matrix of SDF information ready for prediction.

        Examples
        --------
        See Example notebooks for use.
        """
        # calculate sequence info
        if self.is_loaded():
            _intdf, _selector, _sel_col = self._calculate_seq_info(X, regex)
            _new_index = pd.Index(_intdf.loc[_selector, _sel_col].values)
            return _intdf.loc[_selector, self.SDF.view("~id")].set_index(_new_index)
        else:
            raise ValueError("SDF/Model/Labels are not loaded - call `load()` before fetching.")

    def predict(self, X, regex=False):
        """Predicts protein abundance(s) given appropriate gene, mRNA or protein labels.

        Parameters
        ----------
        X : str or array-like of str (n,)
            Given HGNC, Uniprot, Refseq or Ensembl labels. Must match with something
            in SDFModel.hgnc DataFrame.
        regex : bool, optional, default=False
            If True, X must be a str and accepts regular expression.

        Returns
        -------
        y : float or array-like of float (n,)
            Predicted protein abundances.

        Examples
        --------
        See Example notebooks for use.
        """
        if self.is_loaded():
            _intdf, _selector, _sel_col = self._calculate_seq_info(X, regex)
            # now select from columns and get something we can ram into model.
            _Xr = np.asarray(_intdf.loc[_selector, self.SDF.view("~id")])
            # generate predictions
            ypred = self.model.predict(np.atleast_2d(_Xr))
            # wrap in a pandas.series with the given labels.
            return pd.Series(ypred,
                             index=_intdf.loc[_selector, _sel_col].values)
        else:
            raise ValueError("SDF/Model/Labels are not loaded - call `load()` before predicting.")

    def __enter__(self):
        self.load()
        return self

    def __exit__(*args):
        pass

    def __repr__(self):
        return f"SDFModel(loaded={self.is_loaded()})"
