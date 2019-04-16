============================
How to contribute?
============================

In general, you might find this page about `Contributing to Open Source Projects <https://www.contribution-guide.org/>`_
useful.

Feature request or bug found
-----------------------------
If you want to see a feature implemented in the package, `open a issue <https://help.github.com/en/articles/creating-an-issue>`_. If you already have a implementation,
`open a pull request <https://help.github.com/en/articles/creating-a-pull-request>`_.

If you want me to implement a test/sampler or whatever, it would be great if you could point me at a reference
describing the thing you want to have implemented.

.. note::

    Note that we only support python 3.x. We will make no efforts to support
    python 2.x.


.. note::

    At this stage I would be particulary happy about suggestions for better API design.
    During testing I noticed that some function names are probably not the best ones one could
    chose. I would be happy over all suggestions!

Code contributions
------------------
If you want to contribute code, please follow a few little guidelines:

* Try to keep `PEP-8 <https://www.python.org/dev/peps/pep-0008/>`_ in mind. Optimally, use a a tool like
  `yapf <https://github.com/google/yapf>`_ and a linter
  and/or a good IDE like `PyCharm <https://www.jetbrains.com/pycharm/>`_ or simply install the pre-commit hooks with
  ::

    pre-commit install

* Use `Google-Style docstrings <https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html>`_
* Please write unittests.
* I think that type annotations are quite useful (it is e.g. a lot easier to keep the type annotations
  updated than the complete docstrings or documentation), please try to use them as well.
