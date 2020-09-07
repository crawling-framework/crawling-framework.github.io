==================
Installation guide
==================

Requirements
------------

* python version 3
* GCC compiler
* cmake

For MacOS additional:

* brew

Install
-------

For Linux
~~~~~~~~~

* Install all needed python libraries (from project folder)::

   pip3 install -r requirements.txt

* Install `SNAP <https://snap.stanford.edu/snap/index.html>`_ C++ (to any directory)::

   git clone https://github.com/snap-stanford/snap.git
   cd snap

In ``Makefile.config`` add ``-fPIC`` compiler option: find a string 
``CXXFLAGS += -O3 -DNDEBUG -fopenmp``
and replace it with
``CXXFLAGS += -O3 -DNDEBUG -fopenmp -fPIC``

Now build it::

   make all

For MacOS
~~~~~~~~~

* Install all needed python libraries (from project folder)::

   brew install -r requirements.txt

* Install `SNAP <https://snap.stanford.edu/snap/index.html>`_ C++ (to any directory)::

   git clone https://github.com/snap-stanford/snap.git
   cd snap
   make all

Set up configurations
---------------------

Copy the file ``config.example`` to ``config`` - this file will contain your specific
flags and paths::

  cp config.example config

SNAP C++ (for cythonic crawlers)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the ``config`` file set variables::

   SNAP_DIR = "/path/to/snap"         # directory with snap built

Put there your path to the installed snap root directory.
NOTE: don't start the path from '~' or it will fail with the g++ option '-I'.

The following steps are optional, they may be use to speed up computations.

Networkit (for fast betweenness and closeness estimations)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Approximate centralities computing can significantly reduce computation time for large
graphs. Set the corresponding variable to toggle `Networkit <https://networkit.github.io/>`_
library usage (for betweenness and closeness)::

   USE_NETWORKIT = True                 # Use networkit library for approximate centrality calculation

Ligra (for fast eccentricity estimations)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You may install `Ligra <https://github.com/jshun/ligra>`_ framework to use approximate
eccentricity algorithm for large graphs::

   git clone https://github.com/jshun/ligra.git
   cd ligra/apps/eccentricity
   export CILK=" "
   make all
   cd ../../utils
   make all

Set corresponding variables in config file::

   USE_LIGRA = True                     # Use Ligra library for approximate centrality calculation
   LIGRA_DIR = "/path/to/ligra"         # directory with Ligra built

VK messages
~~~~~~~~~~~

You may set your VK account id to get messages from crawler runner (`CrawlerHistoryRunner`).
It reports when computations are complete or when errors occur::

   VK_ID = "00000000"                   # VK id to send info messages

You should allow the `vk_bot` to send messages to you (TODO how?).
