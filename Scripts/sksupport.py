"""
Miscallaenous ml functions to help out with model running.
"""


def adjusted_r2(y_true, y_pred, n, p):
    """Computes the adjusted r2 metric including samples and dimensions."""
    from sklearn.metrics import r2_score
    r2 = r2_score(y_true, y_pred)
    return 1. - (1. - r2) * ((n - 1.) / (n - p - 1.))


def make_ml_ready(df, xcols, ycol, test_size=0.15, zscore_x=False,
                  random_state=None, pprint=False):
    """Prepares data to be ML ready by dropping by ycol, and splitting
    into training and testing/holdout subsets.

    Parameters
    ----------
    df : pandas.DataFrame
    xcols : str or list of str
        Names of columns
    ycol : str
        Target name
    test_size : float, default=0.15
        Proportion of data to hold out. If set to None, will not split data, justs returns X,Y
    zscore_x : bool, default=False
        Whether to standardize x
    random_state : int, default=None
        Sets the seed if nonnull
    pprint : bool, default=False
        Pretty print splitting as debug

    Returns
    -------
    X, Y : ndarray(n, p), ndarray(n,1) if test_size is None
    X_train, X_test, y_train, y_test: ndarray(m, p), ndarray(k, p), ndarray(m,1), ndarray(k,1) if test_size is float
    """
    # convert xcols to list if not already
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import scale
    import warnings

    if isinstance(xcols, str):
        joint_c = [xcols, ycol]
    else:
        joint_c = list(xcols) + [ycol]
    df_main = df[joint_c].dropna()

    # perform sample size change check
    if df_main.shape[0] < (df.shape[0] * .25):
        warnings.warn("more than 75% of ml data samples was dropped ({}->{}".format(
            df.shape[0], df_main.shape[0]), UserWarning)
    # perform N << P check
    if df_main.shape[1] > df_main.shape[0]:
        warnings.warn("dataset has P >> N problem (p={}, n={})".format(
            df_main.shape[1], df_main.shape[0]), UserWarning)

    if isinstance(xcols, str):
        X, Y = np.asarray(df_main[xcols]).reshape(-1, 1), np.asarray(df_main[ycol])
    elif isinstance(xcols, (list, tuple)) and len(xcols) == 1:
        X, Y = np.asarray(df_main[xcols[0]]).reshape(-1, 1), np.asarray(df_main[ycol])
    else:
        X, Y = np.asarray(df_main[list(xcols)]), np.asarray(df_main[ycol])

    if zscore_x:
        X = scale(X)

    # create validation set
    if test_size is not None:
        train_size = int(df_main.shape[0] * (1. - test_size))
        # check that we have a large enough sample
        if train_size <= 50:
            warnings.warn("training size of dataset is {} <= 50; not recommended".format(
                train_size), UserWarning)
        # returns X_train, X_val, y_train, y_val
        split = train_test_split(X, Y, test_size=test_size, random_state=random_state)

        if pprint:
            print("Splitting DF to {}, {} ({}% split), seed={}".format(split[0].shape, split[1].shape, test_size*100,
                                                                       random_state))
        return split
    else:
        return X, Y


def plot_parameter_tune(gs, scoring: str = "r2", y_data=None, score_name: str = "test_score"):
    """Plots the pairwise tuning of every argument."""
    from itertools import combinations
    import re
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib
    from scipy.stats import zscore

    # fetch parameter arguments
    accept_scoring = {'r2', 'rmse', 'mse'}
    known_log_names = {'alpha', 'lambda', 'learning_rate', 'C'}
    scoring_formatter = {"r2": r"$r^2$", "rmse": "RMSE", "mse": "MSE"}
    # check that gs contains the attributes we want
    if not hasattr(gs, "cv_results_"):
        raise ValueError("gs parameter must contain 'cv_results_' attribute.")
    if not hasattr(gs, "best_params_"):
        raise ValueError("gs parameter must contain 'best_params_' attribute.")

    ym, ysd = "mean_" + score_name, "std_" + score_name
    cvr = gs.cv_results_
    ms = matplotlib.rcParams['lines.markersize']
    # if cv_results_ is dict, use keys
    if isinstance(cvr, dict):
        param_names = [s for s in cvr.keys() if re.search("^param_", s)]
    else:
        param_names = [s for s in cvr.dtype.fields if re.search("^param_", s)]
    # generate combinations if we have more than 1 parameter.

    if len(param_names) > 1:
        combs = list(combinations(param_names, 2))
        n_combs = len(combs)
        # make figure
        fs = (8, 5) if n_combs == 1 else (3.5 * n_combs, 2. + (n_combs / 2))
        fig, ax = plt.subplots(ncols=n_combs, figsize=fs)
        if n_combs == 1:
            ax = [ax]
        # compute optimal subset based on scorer
        if scoring == 'r2':
            # eliminate samples < 0
            include_samp = cvr[ym] >= 0
        elif scoring == 'rmse':
            if y_data is not None:
                include_samp = cvr[ym] >= -(np.nanmean(y_data) - (np.nanstd(y_data) * 4))
            else:
                include_samp = np.ones_like(cvr[ym], dtype=np.bool)
        elif scoring == 'mse':
            if y_data is not None:
                include_samp = cvr[ym] >= -np.power(np.nanmean(y_data) - (np.nanstd(y_data) * 4), 2)
            else:
                include_samp = np.ones_like(cvr[ym], dtype=np.bool)
        else:
            raise ValueError(f"scoring '{scoring}' must be in {accept_scoring}")

        has_missing = np.sum(include_samp) < cvr[ym].shape[0]
        # iterate over combinations and plot
        for i, (a, b) in enumerate(combs):
            a_clean = a.replace("param_", "")
            b_clean = b.replace("param_", "")
            # plot missing
            if has_missing:
                ax[i].scatter(cvr[a][~include_samp],
                              cvr[b][~include_samp],
                              c="gray",
                              s=ms,
                              marker='o',
                              alpha=.5)
            # plot good samples
            # markersize modified by standard deviation
            new_sizes = (ms ** 2) + zscore(cvr[ysd][include_samp])

            cax = ax[i].scatter(cvr[a][include_samp],
                                cvr[b][include_samp],
                                c=cvr[ym][include_samp],
                                s=new_sizes,
                                marker='o',
                                cmap="plasma_r")
            # plot best
            ax[i].scatter([gs.best_params_[a_clean]], [gs.best_params_[b_clean]], marker='x', color='r', s=50)
            if a_clean in known_log_names:
                ax[i].set_xscale("log")
            if b_clean in known_log_names:
                ax[i].set_yscale("log")
            ax[i].set_xlabel(a_clean)
            ax[i].set_ylabel(b_clean)

        fig.colorbar(cax, ax=ax[-1])
        fig.tight_layout()
    else:
        fig, ax = plt.subplots(figsize=(8, 5))
        p = param_names[0]
        p_clean = p.replace("param_", "")
        # plot as parameter vs score.
        _order = np.argsort(cvr[p])
        # plot using order
        # handle object typing by checking type
        if isinstance(cvr[p][0], (float, int)):
            _ordered_params = np.asarray(cvr[p][_order]).astype(float)
        else:
            _ordered_params = np.asarray(cvr[p][_order])

        _data_mean = np.asarray(cvr[ym][_order])
        _data_sd = np.asarray(cvr[ysd][_order])

        ax.plot(_ordered_params, _data_mean, 'o-')
        # fill error using between
        ax.fill_between(_ordered_params, _data_mean + _data_sd,
                        _data_mean - _data_sd, alpha=.2)
        # best
        ax.scatter([gs.best_params_[p_clean]], [gs.best_score_], marker='x', color='r', s=50)
        if p_clean in known_log_names:
            ax.set_xscale("log")
        ax.set_xlabel(p_clean)
        ax.set_ylabel(scoring_formatter[scoring])
        fig.tight_layout()