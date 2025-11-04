.. lmpipe documentation master file

Welcome to LMPipe's documentation!
===================================

LMPipe is a flexible framework for running landmark estimation pipelines on videos, 
image sequences, and camera streams.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules

Features
--------

* **Multiple Input Types**: Process videos, image sequences, single images, and camera streams
* **Plugin Architecture**: Extend with custom estimators via entry points
* **Flexible Output**: Save landmarks in various formats (.npy, .csv, .json)
* **Parallel Processing**: Batch and sample-level parallelization support
* **Rich CLI**: Command-line interface with progress tracking

Quick Start
-----------

Installation::

    pip install lmpipe

Basic usage::

    # Process a video file
    lmpipe input.mp4 output/ --pose mediapipe

    # Batch processing
    lmpipe videos/ output/ --holistic mediapipe --max-workers 4

Programmatic usage::

    from lmpipe.interface import LMPipeInterface
    from lmpipe.estimator.holistic import HolisticEstimator

    estimator = HolisticEstimator()
    interface = LMPipeInterface(estimator)
    interface.run('input.mp4', 'output/')

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
