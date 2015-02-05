"""
Test the decoder module
"""
# Author: Yannick Schwartz
# License: simplified BSD

import itertools

import numpy as np
import nibabel

from nose.tools import assert_equal, assert_true, assert_raises
from sklearn.utils import check_random_state
from sklearn.cross_validation import ShuffleSplit
from sklearn.metrics import accuracy_score
from sklearn.base import is_classifier

from ..decoder import Decoder, ESTIMATOR_CATALOG
from ..._utils.testing import generate_fake_fmri
from ...input_data import NiftiMasker
from sklearn.metrics import f1_score


def test_decoder_masking():
    # Generate simple data
    fmri, mask, y = generate_fake_fmri(
        length=10, n_blocks=2, block_size=3)

    # with given mask
    decoder = Decoder(mask=mask)
    decoder.fit(fmri, y)
    assert_true(hasattr(decoder, 'coef_'))

    # with no given mask
    decoder = Decoder()
    decoder.fit(fmri, y)
    assert_true(hasattr(decoder, 'coef_'))

    # with given masker
    masker = NiftiMasker(mask)
    decoder = Decoder(mask=masker)
    decoder.fit(fmri, y)
    assert_true(hasattr(decoder, 'coef_'))


def test_decoder_classification():
    random_state = 10

    # Generate multi-class fake fMRI data
    fmri, mask, y = generate_fake_fmri(
        length=15, n_blocks=3, block_size=4, block_type='classification')

    # Test multi-class and binary classification
    classifiers = [clf for clf in ESTIMATOR_CATALOG
                   if is_classifier(ESTIMATOR_CATALOG[clf])]
    cross_val = [None, ShuffleSplit(y.size, random_state=random_state)]
    screening_percentile = [None, 30]
    scorings = ['accuracy', 'f1', 'r2']

    for select, cv in itertools.product(screening_percentile, cross_val):
        decoder = Decoder(cv=cv,
                          screening_percentile=select,
                          n_jobs=1)
        decoder.fit(fmri, y)
        classes = np.unique(y)

        assert_true(hasattr(decoder, 'cv_y_pred_'))
        assert_true(hasattr(decoder, 'cv_y_true_'))
        assert_true(hasattr(decoder, 'cv_params_'))
        assert_true(hasattr(decoder, 'coef_'))
        assert_true(hasattr(decoder, 'coef_img_'))
        assert_true(decoder.is_classification_)
        assert_equal(sorted(decoder.coef_img_.keys()), sorted(classes))

    fmri, mask, y = generate_fake_fmri(
        length=15, n_blocks=1, block_size=4, block_type='classification')

    # Test decoder for different classifiers
    for classifier in classifiers:
        decoder = Decoder(estimator=classifier)
        decoder.fit(fmri, y)
        decoder.score(fmri, y)

    # Use class names instead of numbers
    y_map = {
        0: 'class_one',
        1: 'class_two',
    }
    y = np.array([y_map.get(v) for v in y])

    # Test scoring methods
    for scoring in scorings:
        if scoring == 'accuracy':
            decoder = Decoder(scoring=scoring)
            decoder.fit(fmri, y)
            decoder.score(fmri, y)
        elif scoring == 'f1':
            # requires a pos_label
            decoder = Decoder(scoring=scoring)
            assert_raises(ValueError, decoder.fit, niimgs=fmri, y=y)

            # requires a valid pos_label
            decoder = Decoder(scoring=scoring, pos_label='data_frame')
            assert_raises(ValueError, decoder.fit, niimgs=fmri, y=y)

            # class_one is valid
            decoder = Decoder(scoring=scoring, pos_label='class_one')
            decoder.fit(fmri, y)
            decoder.score(fmri, y)

            # class_two is valid
            decoder = Decoder(scoring=scoring, pos_label='class_two')
            decoder.fit(fmri, y)
            decoder.score(fmri, y)
        else:
            # Check that r2 scoring raises an error for classification
            decoder = Decoder(scoring=scoring)
            assert_raises(ValueError, decoder.fit, niimgs=fmri, y=y)
            continue

    # Test _check_feature selection
    decoder = Decoder(screening_percentile=101)
    assert_raises(ValueError, decoder.fit, niimgs=fmri, y=y)


def test_decoder_regression():
    # Generate fake fMRI data for regression
    fmri, mask, y = generate_fake_fmri(
        length=15, n_blocks=3, block_size=4, block_type='regression')

    # Test regression
    regrs = [clf for clf in ESTIMATOR_CATALOG
             if not is_classifier(ESTIMATOR_CATALOG[clf])]

    for regr in regrs:
        decoder = Decoder(estimator=regr)
        decoder.fit(fmri, y)
        assert_true(hasattr(decoder, 'coef_img_'))
        assert_equal(decoder.coef_img_.keys(), ['beta'])
        decoder.predict(fmri)
        assert_true(not decoder.is_classification_)
