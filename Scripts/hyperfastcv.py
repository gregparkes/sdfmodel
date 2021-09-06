"""File for creating hyperfast gridsearching over a scikit-learn hyperparameter domain."""

import itertools as it
import numpy as np

from sklearn.model_selection import cross_validate
from joblib import cpu_count, Parallel, delayed
from functools import partial
from scipy import stats

from sklearn.base import clone
from abc import ABCMeta
from sklearn.metrics import make_scorer, r2_score


def _adjusted_r2(y_true, y_pred, n, p):
    """Computes the adjusted r2 metric including samples and dimensions."""
    r2 = r2_score(y_true, y_pred)
    return 1. - (((1. - r2) * (n - 1.)) / (n - p - 1.))


def _has_tqdm():
    try:
        from tqdm import tqdm
        return True
    except ImportError:
        return False


def _retuple(tup):
    """Reshapes a list of tuples into a set of arguments that are mappable."""
    # length must be greater than 1
    L = len(tup[0])
    # firstly assert that every tuple is of the same length
    assert all(map(lambda t: len(t) == L, tup)), "not all tuples are same length"
    # now reshape according to this length
    return [tuple(map(lambda t: t[i], tup)) for i in range(L)]


class HyperFastCV:
    """Performing HyperParameter GridSearch tuning on subsets."""
    def __init__(self, estimator, param_grid, n_trials=10,
                 chunk=0.2, cv=None, verbose=0,
                 n_jobs=None, use_n_estimators=False):
        self.estimator = estimator
        self.param_grid = param_grid
        self.cv = 5 if cv is None else cv
        self.chunk = chunk
        self.verbose = verbose
        self.n_trials = n_trials
        self.n_jobs = 1 if n_jobs is None else n_jobs
        self.param_keys = list(self.param_grid.keys())
        self._param_cv_keys = list(map(lambda s: "param_" + s, self.param_keys))
        self.use_n_estimators = use_n_estimators
        self._param_counts = {}
        self.scoring = "r2"

    def _sample_param(self, k, v):
        if hasattr(v, "rvs"):
            return v.rvs()
        else:
            probs = self._param_counts[k] / np.sum(self._param_counts[k])
            # is a list, tuple of parameters
            _samp = np.random.choice(v, p=probs)
            # update param probs to add where indexes match.
            self._param_counts[k][np.argwhere(v != _samp).flatten()] += 1
            return _samp

    def _get_sample_indices(self, n, p):
        # now compute indices
        if np.isclose(self.chunk, 1.):
            _sample_size = n
            _indices = np.arange(n)
        else:
            # make sampler
            if hasattr(self.estimator, "n_estimators") and self.use_n_estimators:
                _sample_size = np.random.randint(5, 20)

            else:
                if isinstance(self.chunk, float):
                    n_part = int(n * self.chunk)
                    _lower, _upper = max(n_part, p), min(n_part * 2, n - 1)
                    if _lower < _upper:
                        _sample_size = np.random.randint(_lower, _upper)
                    else:
                        _sample_size = n
                elif isinstance(self.chunk, (list, tuple)):
                    n_lower, n_upper = self.chunk
                    if n_upper <= n_lower:
                        raise ValueError("chunk tuple {} upper must be > lower".format(self.chunk))

                    _sample_size = np.random.randint(max(int(n * n_lower), p), min(int(n * n_upper), n - 1))
                else:
                    raise ValueError(f"type of chunk '{type(self.chunk)}' not recognized")

            _indices = np.random.choice(n, size=_sample_size, replace=False)

        return _sample_size, _indices

    def _make_iterator(self):
        # tqdm is optional, we wrap our iterable with it if installed.
        if _has_tqdm():
            from tqdm import tqdm
            return tqdm(range(self.n_trials), position=0)
        else:
            return range(self.n_trials)

    def _fill_results(self, i, _sample_size, nx, ny, _cv_params, _indices):
        # sample a parameter from the distributions.
        _param_step = {k: self._sample_param(k, v) for k, v in self.param_grid.items()}

        if hasattr(self.estimator, "n_estimators") and self.use_n_estimators:
            # add on parameter in param step.
            _cv = cross_validate(clone(self.estimator.set_params(n_estimators=_sample_size,
                                                                 **_param_step)),
                                 nx, ny, **_cv_params)
        else:
            _cv = cross_validate(clone(self.estimator.set_params(**_param_step)),
                                 nx[_indices], ny[_indices], **_cv_params)

        # set results
        self.cv_results_['mean_test_score'][i] = _cv['test_score'].mean()
        self.cv_results_['std_test_score'][i] = _cv['test_score'].std()
        self.cv_results_['mean_train_score'][i] = _cv['train_score'].mean()
        self.cv_results_['std_train_score'][i] = _cv['train_score'].std()
        self.cv_results_['mean_fit_time'][i] = _cv['fit_time'].mean()
        self.cv_results_['std_fit_time'][i] = _cv['fit_time'].std()
        # iterate over param args and fill in
        for k in _param_step:
            self.cv_results_["param_" + k][i] = _param_step[k]

    def is_fitted(self):
        """Determines whether the model has been fitted or not."""
        return hasattr(self, "cv_results_")

    @property
    def best_index_(self):
        """Returns the top index."""
        if self.is_fitted():
            return np.argmax(self.cv_results_['mean_test_score'])
        else:
            raise ValueError("model not fitted!")

    @property
    def best_score_(self):
        """Returns the top score."""
        if self.is_fitted():
            return self.cv_results_['mean_test_score'][self.best_index_]
        else:
            raise ValueError("model not fitted!")

    @property
    def best_params_(self):
        """Returns the best parameter set."""
        if self.is_fitted():
            return {k: self.cv_results_["param_" + k][self.best_index_] for k in
                        self.param_grid}
        else:
            raise ValueError("model not fitted!")

    @property
    def best_time_(self):
        """Returns the time for best score."""
        if self.is_fitted():
            return self.cv_results_['mean_fit_time'][self.best_index_]
        else:
            raise ValueError("model not fitted!")

    def fit(self, X, y, **fit_kwargs):
        """Run fit with all sets of parameters.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and n_features is the number of features.
        y : array-like of shape (n_samples, n_output) or (n_samples,), default=None
            Target relative to X for classification or regression; None for unsupervised learning.
        """
        # estimate rough time length from size of the data.
        n, p = X.shape
        # if n_estimators is a valid attribute of the estimator, we will use this instead.
        if p > n:
            raise ValueError("p >> n situations cannot be allowed.")
        nx = X.copy()
        ny = y.copy()

        # assign param probabilities
        for k in self.param_grid.keys():
            if hasattr(self.param_grid[k], "rvs"):
                # it's a probability distribution, all samples equal
                self._param_counts[k] = None
            else:
                # we then update these probabilities everytime we pick one.
                n_params_for_k = len(self.param_grid[k])
                self._param_counts[k] = np.ones(n_params_for_k, dtype=np.float64)

        # create a scoring function based on data size
        if p < 5:
            scorer = r2_score
        else:
            scorer = partial(_adjusted_r2, n=n, p=p)
            scorer.__name__ = "adj_r2"
        # unchanging params.
        _cv_params = dict(scoring=make_scorer(scorer), cv=self.cv,
                          return_train_score=True, fit_params=fit_kwargs)

        _sample_size, _indices = self._get_sample_indices(n, p)
        itr = self._make_iterator()

        _param_args = [("param_"+k, type(self._sample_param(k, v))) for k, v in self.param_grid.items()]

        # make a numpy matrix.
        self.cv_results_ = np.zeros(self.n_trials, dtype=[
            ('mean_fit_time', 'f8'), ('mean_train_score', 'f8'), ('mean_test_score', 'f8'),
            ('std_train_score', 'f8'), ('std_test_score', 'f8'), ("std_fit_time", 'f8'),
            ("N", 'i8')
        ] + _param_args)
        self.cv_results_['N'] = _indices.shape[0]

        # iterate here.
        if self.n_jobs > 1:
            ncpu = self.n_trials if self.n_trials < cpu_count() else self.n_jobs
            Parallel(ncpu, prefer="threads")(delayed(self._fill_results)(i, _sample_size, nx, ny, _cv_params, _indices) for i in itr)
        else:
            _ = [self._fill_results(i, _sample_size, nx, ny, _cv_params, _indices) for i in itr]
        return self
