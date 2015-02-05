"""High-level decoding object that exposes standard classification
and regression strategies such as SVM, LogisticRegression and Ridge,
with optional feature selection, and integrated parameter selection.
"""
# Authors : Yannick Schwartz
#
# License: simplified BSD


import itertools
import warnings
import copy

import numpy as np

from sklearn.externals.joblib import Parallel, delayed, Memory
from sklearn.linear_model.ridge import Ridge, RidgeClassifier, _BaseRidge
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVR
from sklearn.preprocessing import LabelBinarizer
from sklearn.feature_selection import SelectPercentile, SelectKBest
from sklearn.feature_selection import f_classif, f_regression
from sklearn.svm.bounds import l1_min_c
from sklearn.metrics import make_scorer
from sklearn.metrics import r2_score, f1_score, precision_score, recall_score
from sklearn.metrics.scorer import check_scoring
from sklearn.grid_search import ParameterGrid
from sklearn.base import BaseEstimator
from sklearn.base import is_classifier
from sklearn import clone

try:
    from sklearn.utils import check_arrays as check_array
except ImportError as e:
    from sklearn.utils import check_array

try:
    # scikit-learn < 0.17
    from sklearn.cross_validation import _check_cv as check_cv
except ImportError as e:
    # scikit-learn >= 0.17
    from sklearn.cross_validation import check_cv

from ..input_data import NiftiMasker, MultiNiftiMasker


ESTIMATOR_CATALOG = dict(
    svc_l1=LinearSVC(penalty='l1', dual=False, class_weight='auto'),
    svc_l2=LinearSVC(penalty='l2', class_weight='auto'),
    svc=LinearSVC(penalty='l2', class_weight='auto'),
    logistic_l1=LogisticRegression(penalty='l1', class_weight='auto'),
    logistic_l2=LogisticRegression(penalty='l2', class_weight='auto'),
    logistic=LogisticRegression(penalty='l2', class_weight='auto'),
    ridge_classifier=RidgeClassifier(class_weight='auto'),
    ridge_regression=Ridge(),
    svr=SVR(kernel='linear'),
)

REQUIRES_POS_LABEL = [f1_score, precision_score, recall_score]


class Decoder(BaseEstimator):
    """Popular classification and regression strategies for neuroimgaging.

    The `Decoder` object supports classification and regression methods.
    It implements a model selection scheme that averages the best models
    within a cross validation loop. The resulting average model is the
    one used as a classifier or a regressor. The `Decoder` object also
    leverages the `NiftiMaskers` to provide a direct interface with the
    nifti files on disk.

    Parameters
    -----------
    estimator : str, optional
        The estimator to choose among: 'svc', 'svc_l1', 'logistic',
        'logistic_l1', 'ridge_classifier', 'ridge_regression',
        and 'svr'. Defaults to 'svc'.

    mask: filename, NiImage, NiftiMasker, or MultiNiftiMasker, optional
        Mask to be used on data. If an instance of masker is passed,
        then its mask will be used. If no mask is given,
        it will be computed automatically by a masker with default
        parameters.

    cv : cross-validation generator, optional
        A cross-validation generator. If None, a 3-fold cross
        validation is used for regression or 3-fold stratified
        cross-validation for classification.

    param_grid : dict of string to sequence, or sequence of such
        The parameter grid to explore, as a dictionary mapping estimator
        parameters to sequences of allowed values.

        An empty dict signifies default parameters.

        A sequence of dicts signifies a sequence of grids to search, and is
        useful to avoid exploring parameter combinations that make no sense
        or have no effect. See scikit-learn documentation for more information.

    select_features: int, float, Transformer or None, optional
        Perform an univariate feature selection based on the Anova F-value for
        the input data. An integer select features according to the k highest
        scores, a float according to a percentile of the highest scores.
        If None no feature selection is performed. A custom feature
        selection may be applied by passing a Transformer object.
        Defaults to .2.

    pos_label: str or None
        The positive class label in case of a binary classification. It is
        used in the case of binary classification, and only the following
        scoring metrics require it: f1, precision, and recall.

    scoring : string or callable, optional
        The scoring strategy to use. See the scikit-learn documentation
        If callable, takes as arguments the fitted estimator, the
        test data (X_test) and the test target (y_test) if y is
        not None.

    smoothing_fwhm: float, optional
        If smoothing_fwhm is not None, it gives the size in millimeters of the
        spatial smoothing to apply to the signal.

    standardize : boolean, optional
        If standardize is True, the time-series are centered and normed:
        their variance is put to 1 in the time dimension.

    target_affine: 3x3 or 4x4 matrix, optional
        This parameter is passed to image.resample_img. Please see the
        related documentation for details.

    target_shape: 3-tuple of integers, optional
        This parameter is passed to image.resample_img. Please see the
        related documentation for details.

    mask_strategy: {'background' or 'epi'}, optional
        The strategy used to compute the mask: use 'background' if your
        images present a clear homogeneous background, and 'epi' if they
        are raw EPI images. Depending on this value, the mask will be
        computed from masking.compute_background_mask or
        masking.compute_epi_mask. Default is 'background'.

    memory: instance of joblib.Memory or string
        Used to cache the masking process.
        By default, no caching is done. If a string is given, it is the
        path to the caching directory.

    memory_level: integer, optional
        Rough estimator of the amount of memory used by caching. Higher value
        means more memory for caching.

    n_jobs : int, optional. Default is -1.
        The number of CPUs to use to do the computation. -1 means
        'all CPUs'.

    verbose : int, optional
        Verbosity level. Default is False
    """

    def __init__(self, estimator='svc', mask=None, cv=None, param_grid=None,
                 select_features=.2, pos_label=None, scoring=None,
                 smoothing_fwhm=None, standardize=True, target_affine=None,
                 target_shape=None, mask_strategy='background',
                 memory=Memory(cachedir=None), memory_level=0,
                 n_jobs=1, verbose=False,
                 ):
        self.estimator = estimator
        self.mask = mask
        self.cv = cv
        self.param_grid = param_grid
        self.select_features = select_features
        self.pos_label = pos_label
        self.scoring = scoring
        self.smoothing_fwhm = smoothing_fwhm
        self.standardize = standardize
        self.target_affine = target_affine
        self.target_shape = target_shape
        self.mask_strategy = mask_strategy
        self.memory = memory
        self.memory_level = memory_level
        self.n_jobs = n_jobs
        self.verbose = verbose

    def fit(self, niimgs, y):
        """Fit the decoder

        Parameters
        ----------
        niimgs: list of filenames or NiImages
            Data on which the mask must be calculated. If this is a list,
            the affine is considered the same for all.

        y: 1D array-like
           Target variable to predict. Must have exactly as many elements as
           3D images in niimg.

        Attributes
        ----------
        `mask_img_`: NiImage
            Mask computed by the masker object.
        `classes_`: numpy.ndarray
            Classes to predict. For classification only.
        `coef_`: numpy.ndarray, shape=(n_classes, n_features)
            Contains the mean of the models weight vector across
            fold for each class.
        `std_coef_`: numpy.ndarray, shape=(n_classes, n_features)
            Contains the standard deviation of the models weight vector across
            fold for each class.
        `coef_img_`: dict of NiImage
            Dictionary containing `coef_` with class names as keys,
            and `coef_` transformed in NiImages as values. In the case
            of a regression, it contains a single NiImage at the key 'beta'.
        `std_coef_img_`: dict of NiImage
            Dictionary containing `std_coef_` with class names as keys,
            and `coef_` transformed in NiImages as values. In the case
            of a regression, it contains a single NiImage at the key 'beta'.
        `cv_y_true_` : numpy.ndarray
            Ground truth labels for left out samples in inner cross validation.
        `cv_y_pred_` : numpy.ndarray
            Predicted labels for left out samples in inner cross validation.
        `cv_params_`: dict of lists
            Best point in the parameter grid for each tested fold
            in the inner cross validation loop.
        """
        # Setup memory, parallel and masker
        if isinstance(self.memory, basestring) or self.memory is None:
            self.memory = Memory(cachedir=self.memory, verbose=self.verbose)

        parallel = Parallel(n_jobs=self.n_jobs, verbose=self.verbose,
                            pre_dispatch='n_jobs')

        self.masker_ = _check_masking(self.mask, self.smoothing_fwhm,
                                      self.target_affine, self.target_shape,
                                      self.standardize, self.mask_strategy,
                                      self.memory, self.memory_level)

        # Fit masker
        if hasattr(self.masker_, 'mask_img_'):
            self.masker_.fit(niimgs)
        else:
            self.masker_.fit()
        self.mask_img_ = self.masker_.mask_img_

        # Load data and target
        X = self.masker_.transform(niimgs)
        X = np.vstack(X) if isinstance(X, tuple) else X
        y, = check_array(y)


        # Setup model
        if not isinstance(self.estimator, basestring):
            warnings.warn('Use a custom estimator at your own risk.')
        estimator = ESTIMATOR_CATALOG.get(self.estimator, self.estimator)
        is_classification_, self.is_binary_, classes_, classes_to_predict = \
            _check_estimation(estimator, y, self.pos_label)
        if classes_ is not None:
            self.classes_ = classes_
        self.is_classification_ = is_classification_

        # Raise an error early if something is wrong with the scoring
        scorer, scoring = _check_scorer(self, self.scoring, self.pos_label, y)

        # Setup cv
        cv = check_cv(self.cv, X, y, classifier=is_classification_)

        # Train all labels in all folds
        results = parallel(delayed(_parallel_estimate)(
            estimator, X, y, train, test, self.param_grid,
            pos_label, is_classification_, scoring,
            self.select_features, self.verbose)
            for pos_label, (train, test)
            in itertools.product(classes_to_predict, cv))

        # Gather results
        coefs = {}
        intercepts = {}
        cv_pred = {}
        cv_true = {}
        self.cv_params_ = {}
        for c, best_coef, best_intercept, best_y, best_params in results:
            coefs.setdefault(c, []).append(best_coef)
            intercepts.setdefault(c, []).append(best_intercept)
            self.cv_params_.setdefault(c, {})
            for k in best_params:
                self.cv_params_[c].setdefault(k, []).append(best_params[k])
            cv_pred.setdefault(c, []).append(best_y['y_pred'])
            cv_true.setdefault(c, []).append(best_y['y_true'])

            if self.is_binary_:
                other_class = np.setdiff1d(
                    self.classes_, classes_to_predict)[0]
                cv_pred.setdefault(other_class, []).append(best_y['inverse'])
                coefs.setdefault(other_class, []).append(-best_coef)
                intercepts.setdefault(other_class, []).append(-best_intercept)

        if is_classification_:
            classes = self.classes_

            y_probs = np.array([np.vstack(cv_pred[c]).T for c in classes]).T
            self.cv_y_pred_ = []
            for fold in np.arange(y_probs.shape[0]):
                self.cv_y_pred_.append(self.classes_[
                    np.argmax(y_probs[fold], axis=1)])
            self.cv_y_true_ = cv_true[cv_true.keys()[0]]
        else:
            classes = classes_to_predict
            self.cv_y_pred_ = np.vstack([
                np.hstack(cv_pred[c]) for c in classes]).T.ravel()
            self.cv_y_true_ = np.vstack([
                np.hstack(cv_true[c]) for c in classes]).T.ravel()
            self.cv_params_['beta'] = self.cv_params_.pop(None)

        self.coef_ = np.vstack([np.mean(coefs[c], axis=0) for c in classes])
        self.std_coef_ = np.vstack([np.std(coefs[c], axis=0) for c in classes])
        self.intercept_ = np.hstack([np.mean(intercepts[c], axis=0)
                                     for c in classes])
        self.coef_img_ = {}
        self.std_coef_img_ = {}
        for c, coef, std in zip(classes, self.coef_, self.std_coef_):
            c = 'beta' if c is None else c
            self.coef_img_[c] = self.masker_.inverse_transform(coef)
            self.std_coef_img_[c] = self.masker_.inverse_transform(std)

    def decision_function(self, niimgs):
        """Provide prediction values for new X which can be turned into
        a label by thresholding
        """
        X = self.masker_.transform(niimgs)
        if isinstance(X, tuple):
            X = np.vstack(X)
        X_view = X.view()
        X_view.shape = (X.shape[0], -1)
        decision_values = np.dot(X_view, self.coef_.T) + self.intercept_
        return decision_values

    def predict(self, niimgs):
        """Predict a label for all X vectors indexed by the first axis"""
        decision_values = self.decision_function(niimgs)
        if self.is_classification_:
            decisions = decision_values.argmax(axis=1)
            decision_labels = self.classes_[decisions]
            return decision_labels
        return decision_values

    def score(self, niimgs, y):
        scorer, _ = _check_scorer(self, self.scoring, self.pos_label, y)
        return scorer(self, niimgs, y)


def _parallel_estimate(estimator, X, y, train, test, param_grid,
                       pos_label, is_classification, scoring,
                       select_features=None, verbose=0):
    """Find the best estimator for a fold within a job."""

    if is_classification and pos_label is None:
        raise warnings.warn(
            'It seems like you have a classification task'
            'but you did not specify which label you are trying to '
            'predict: y=%s' % y)

    y_true = y[test]
    if pos_label is not None:
        label_binarizer = LabelBinarizer(pos_label=1, neg_label=-1)
        y = label_binarizer.fit_transform(y == pos_label).ravel()

    scorer = check_scoring(estimator, scoring)
    select_features = _check_feature_selection(select_features,
                                               is_classification)
    if select_features is not None:
        X_train = select_features.fit_transform(X[train], y[train])
        X_test = select_features.transform(X[test])
    else:
        X_train = X[train]
        X_test = X[test]

    param_grid = _check_param_grid(estimator, X_train, y[train], param_grid)

    best_score = None
    for param in ParameterGrid(param_grid):
        estimator = clone(estimator).set_params(**param)
        estimator.fit(X_train, y[train])

        if is_classification:
            if hasattr(estimator, 'predict_proba'):
                y_prob = estimator.predict_proba(X_test)
                y_prob = y_prob[:, 1]
                inverse_prob = 1 - y_prob
            else:
                decision = estimator.decision_function(X_test)
                if decision.ndim == 2:
                    y_prob = decision[:, 1]
                    inverse_prob = np.abs(decision[:, 0])
                else:
                    y_prob = decision
                    inverse_prob = -decision
            score = scorer(estimator, X_test, y[test])
            if np.all(estimator.coef_ == 0):
                score = 0
        else:  # regression
            y_prob = estimator.predict(X_test)
            score = scorer(estimator, X_test, y[test])
        if best_score is None or score >= best_score:
            best_score = score
            best_coef = estimator.coef_
            best_intercept = estimator.intercept_
            best_y = {}
            best_y['y_pred'] = y_prob
            best_y['y_true'] = y_true
            best_params = param
            if is_classification:
                best_y['inverse'] = inverse_prob

    if select_features is not None:
        best_coef = select_features.inverse_transform(best_coef)
    # if sklearn.__version__ > 1.7
    if isinstance(estimator, SVR):
        best_coef = -best_coef

    return pos_label, best_coef, best_intercept, best_y, best_params


def _check_masking(mask, smoothing_fwhm, target_affine, target_shape,
                   standardize, mask_strategy, memory, memory_level):
    """Setup a nifti masker."""
    # Mask is an image, not a masker
    if not isinstance(mask, (NiftiMasker, MultiNiftiMasker)):
        masker = NiftiMasker(mask=mask,
                             smoothing_fwhm=smoothing_fwhm,
                             target_affine=target_affine,
                             target_shape=target_shape,
                             standardize=standardize,
                             mask_strategy=mask_strategy,
                             memory=memory,
                             memory_level=memory_level, )
    # Mask is a masker object
    else:
        try:
            masker = clone(mask)
        except TypeError as e:
            # Workaround for a joblib bug: in joblib 0.6, a Memory object
            # with cachedir = None cannot be cloned.
            masker_memory = mask.memory
            if masker_memory.cachedir is None:
                mask.memory = None
                masker = clone(mask)
                mask.memory = masker_memory
                masker.memory = Memory(cachedir=None)
            else:
                # The error was raised for another reason
                raise e

        for param_name in ['target_affine', 'target_shape',
                           'smoothing_fwhm', 'mask_strategy',
                           'memory', 'memory_level']:
            if getattr(masker, param_name) is not None:
                warnings.warn('Parameter %s of the masker overriden'
                              % param_name)
                setattr(masker, param_name, locals()[param_name])
    return masker


def _check_param_grid(estimator, X, y, param_grid):
    """Check param_grid and return sensible default if none is given.
    """
    if param_grid is None:
        param_grid = {}
        # define loss function
        if isinstance(estimator, LogisticRegression):
            loss = 'log'
        elif isinstance(estimator, (LinearSVC, _BaseRidge, SVR)):
            loss = 'l2'
        min_c = l1_min_c(X, y, loss=loss) \
            if hasattr(estimator, 'penalty') \
            and estimator.penalty == 'l1' else .5
        param_grid['C'] = np.array([2, 20, 200]) * min_c
        if isinstance(estimator, _BaseRidge):
            param_grid['alpha'] = 1. / (param_grid.pop('C') * 2)
    return param_grid


def _check_feature_selection(select_features, is_classification):
    """Check feature selection method. Turns floats in SelectPercentile
    objects, integers in SelectKBest objects.
    """

    if not is_classification:
        f_test = f_regression
    else:
        f_test = f_classif

    if isinstance(select_features, int):
        if select_features <= 1.:
            raise ValueError("When `select_features` is an integer it "
                             "needs to be greater than one and smaller "
                             "or equal to n_features.")
        return SelectKBest(f_test, select_features)
    elif isinstance(select_features, float):
        if select_features > 1.:
            raise ValueError("When `select_features` is a float it "
                             "should either be smaller or equal to 1.")
        return SelectPercentile(f_test, select_features * 100)
    return select_features


def _check_estimation(estimator, y, pos_label):
    """Check estimation problem in respect to target type."""
    is_classification_ = is_classifier(estimator)
    target_type = y.dtype.kind == 'i' or y.dtype.kind == 'S'
    is_binary = False
    if is_classification_ != target_type:
        warnings.warn(
            'Target seems to be for a %s problem but '
            'chosen estimator is for a %s problem.' % (
                'classification' if target_type else 'regression',
                'classification' if is_classification_ else 'regression'))
    if is_classification_:
        classes_ = classes_to_predict = np.unique(y)
        # If the problem is binary classification we compute only the
        # model for one class and flip the signs for the other class
        if len(classes_) == 2:
            if pos_label is not None and pos_label in classes_:
                classes_to_predict = np.array([pos_label])
            else:
                classes_to_predict = classes_[:1]
            is_binary = True
    else:
        classes_to_predict = [None]
        classes_ = None

    return is_classification_, is_binary, classes_, classes_to_predict


def _check_scorer(estimator, scoring, pos_label, y):
    """Utility function to set up scoring metric.

    Check that metric is valid for the learning problem.
    Make use of pos_label when classification is binary
    and scoring method requires it.
    """
    # Set scoring to a reasonable default
    if scoring is None and estimator.is_classification_:
        scoring = 'accuracy'
    elif scoring is None and not estimator.is_classification_:
        scoring = 'r2'

    # Check that the passed scoring does not raise an Exception
    scorer = check_scoring(estimator, scoring)
    score_func = scorer._score_func if hasattr(scorer, '_score_func') else None

    # Check scoring is for right learning problem
    is_r2 = score_func is r2_score
    if not estimator.is_classification_ and not is_r2:
        raise ValueError('Wrong scoring method `%s` for regression.' % scoring)
    if estimator.is_classification_ and is_r2:
        raise ValueError('Wrong scoring method `%s` '
                         'for classification.' % scoring)

    # Check that pos_label is correctly set if needed
    if (estimator.is_binary_ and np.array(y).dtype.kind == 'S'
            and score_func in REQUIRES_POS_LABEL):

        if pos_label is None:
            raise ValueError('Decoder must be given a pos_label in '
                             'the case of a binary classification '
                             'with `%s` scoring metric.' % scoring)
        elif pos_label not in estimator.classes_:
            raise ValueError(
                'The given pos_label `%s` is not in the target '
                'which contains the classes `%s` and `%s`.' % (
                    pos_label, estimator.classes_[0], estimator.classes_[1]))

    scorer = check_scoring(estimator, scoring)
    if estimator.is_binary_ and scorer._score_func in REQUIRES_POS_LABEL:
        scorer = make_scorer(scorer._score_func,
                             pos_label=pos_label)
    return scorer, scoring
