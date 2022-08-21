# Version  0.1.1

#### New Features

* [#338](https://github.com/pykale/pykale/pull/338): Improve GripNet implementation
* [#339](https://github.com/pykale/pykale/pull/339): Add setup options
* [#340](https://github.com/pykale/pykale/pull/340): Update reading DICOM and marker visualization

#### Code Improvements

* [#341](https://github.com/pykale/pykale/pull/341): Update Colab installation and add notebook hook
* [#342](https://github.com/pykale/pykale/pull/342): Add arguments to visualize and rename examples

#### Documentation Updates

* [#337](https://github.com/pykale/pykale/pull/337): Update GripNet example name and contributing guidelines
* [#343](https://github.com/pykale/pykale/pull/343): Clarify python version supported

# Version  0.1.0

#### New Features

* [#246](https://github.com/pykale/pykale/pull/246): Add MIDA, CoIRLS, distribution plot, and brain example

#### Bug Fixes

* [#322](https://github.com/pykale/pykale/pull/322): Add pre-commit dependency for black and click
* [#330](https://github.com/pykale/pykale/pull/330): Fix problems of tests for Python version 3.7, 3.8 and 3.9

#### Code Improvements

* [#284](https://github.com/pykale/pykale/pull/284): Update DICOM reading and image visualization
* [#320](https://github.com/pykale/pykale/pull/320): Add code scanning
* [#321](https://github.com/pykale/pykale/pull/321): Fix cardiac MRI example visualization number of columns
* [#331](https://github.com/pykale/pykale/pull/331): Update cmr example landmark visualization

#### Documentation Updates

* [#333](https://github.com/pykale/pykale/pull/333): Update docs and readme for 0.1.0 release

# Version  0.1.0rc5

#### New Features

* [#251](https://github.com/pykale/pykale/pull/251): MFSAN support 1D input
* [#273](https://github.com/pykale/pykale/pull/273): Add topk & multitask topk accuracies

#### Bug Fixes

* [#244](https://github.com/pykale/pykale/pull/244): Update getting indicies with torch.where
* [#254](https://github.com/pykale/pykale/pull/254): Fix bugs for upgrading PyTroch-lightning to 1.5
* [#256](https://github.com/pykale/pykale/pull/256) & [#257](https://github.com/pykale/pykale/pull/257): Update for PyTorch 1.10 and Torchvision 0.11.1
* [#286](https://github.com/pykale/pykale/pull/286): Update ipython requirement from <8.0 to <9.0

#### Code Improvements

* [#240](https://github.com/pykale/pykale/pull/240): Refractor the code to save the images instead of opening them at runtime
* [#271](https://github.com/pykale/pykale/pull/271): Fix doc build, improve docstrings and MPCA pipeline fit efficiency
* [#272](https://github.com/pykale/pykale/pull/272): Update progress_bar for PyTorch Lightning & change 'target' abbreviation
* [#283](https://github.com/pykale/pykale/pull/283): "val" in variable names to "valid"

#### Tests

* [#258](https://github.com/pykale/pykale/pull/258): Use pyparsing 2.4.7 in test

#### Documentation Updates

* [#228](https://github.com/pykale/pykale/pull/228): Zenodo json
* [#243](https://github.com/pykale/pykale/pull/243): Clarify PR template
* [#282](https://github.com/pykale/pykale/pull/282): Clarify when to request review and prefer just one label

# Version  0.1.0rc4

#### Code Improvements

* [#218](https://github.com/pykale/pykale/pull/218): Change logger in digits and action examples
* [#219](https://github.com/pykale/pykale/pull/219): Update three notebooks
* [#222](https://github.com/pykale/pykale/pull/222): Add multi source example
* [#224](https://github.com/pykale/pykale/pull/224): Merge all image accesses to a unique API

#### Tests

* [#221](https://github.com/pykale/pykale/pull/221): Add notebook "smoke tests" to CI

#### Documentation Updates

* [#225](https://github.com/pykale/pykale/pull/225): Update readme & fix colab imgaug
* [#229](https://github.com/pykale/pykale/pull/229): Add DOI to readme
* [#235](https://github.com/pykale/pykale/pull/235): Fix typo and hyperlink

# Version  0.1.0rc3

#### New Features

* [#196](https://github.com/pykale/pykale/pull/196): Add Google Drive Download API
* [#197](https://github.com/pykale/pykale/pull/197): Multi domain loader and office data access
* [#210](https://github.com/pykale/pykale/pull/210): Multi-source domain adaptation SOTA

#### Code Improvements

* [#201](https://github.com/pykale/pykale/pull/201): No "extras", only "normal" or "dev" installs

#### Tests

* [#178](https://github.com/pykale/pykale/pull/178): Reduce tests for video
* [#188](https://github.com/pykale/pykale/pull/188): Create download_path directory in conftest.py
* [#189](https://github.com/pykale/pykale/pull/189): Create test_sampler.py and update doc for tests
* [#200](https://github.com/pykale/pykale/pull/200): Nightly test run

#### Documentation Updates

* [#165](https://github.com/pykale/pykale/pull/165): Notebook tutorial for the bindingdb_deepdta example
* [#199](https://github.com/pykale/pykale/pull/199): CMR PAH notebook example
* [#207](https://github.com/pykale/pykale/pull/207): Restructure notebook tutorial docs
* [#212](https://github.com/pykale/pykale/pull/212): Describe use of YAML

#### Other Changes

* [#187](https://github.com/pykale/pykale/pull/187): Add dependabot
* [#205](https://github.com/pykale/pykale/pull/205): Update data dirs

# Version  0.1.0rc2

#### New Features

* [#149](https://github.com/pykale/pykale/pull/149): Add digits notebook with Binder and Colab
* [#151](https://github.com/pykale/pykale/pull/151): Add class subset selection
* [#159](https://github.com/pykale/pykale/pull/159): Add interpret module

#### Code Improvements

* [#132](https://github.com/pykale/pykale/pull/132): Create file download module
* [#138](https://github.com/pykale/pykale/pull/138): Change action_domain_adapter.py to video_domain_adapter.py
* [#144](https://github.com/pykale/pykale/pull/144): Move gait data to pykale/data
* [#157](https://github.com/pykale/pykale/pull/157): Add concord_index calculation into DeepDTA

#### Tests

* [#127](https://github.com/pykale/pykale/pull/127): Add video_access tests
* [#134](https://github.com/pykale/pykale/pull/134): Add tests for image and video CNNs
* [#136](https://github.com/pykale/pykale/pull/136): Add tests for domain adapter
* [#137](https://github.com/pykale/pykale/pull/137): Add tests for csv logger
* [#139](https://github.com/pykale/pykale/pull/139): Add tests for isonet
* [#145](https://github.com/pykale/pykale/pull/145): Add tests for video domain adapter
* [#150](https://github.com/pykale/pykale/pull/150): Add tests for gripnet
* [#156](https://github.com/pykale/pykale/pull/156): Remove empty tests and MNIST test

#### Documentation Updates

* [#133](https://github.com/pykale/pykale/pull/133): Add quote from Kevin@facebook
* [#143](https://github.com/pykale/pykale/pull/143): Add "new feature" group
* [#160](https://github.com/pykale/pykale/pull/160): Update docs w.r.t. CIKM submission

# Version  0.1.0rc1

**Important**: Rename `master` to `main`.

#### Code Improvements

* [#92](https://github.com/pykale/pykale/pull/92): Update action domain adaptation pipeline and modules (big PR)
* [#123](https://github.com/pykale/pykale/pull/123): Merge prep_cmr with image_transform plus tests

#### Tests

* [#104](https://github.com/pykale/pykale/pull/104): Test attention_cnn
* [#107](https://github.com/pykale/pykale/pull/107): Do only CI test multiple python versions on Linux
* [#122](https://github.com/pykale/pykale/pull/122): Test deep_dta

#### Documentation Updates

* [#106](https://github.com/pykale/pykale/pull/106): Update the readmes of docs, examples, and tests
* [#120](https://github.com/pykale/pykale/pull/120): Update PR for changelog, cherry pick, and test re-run tip
* [#121](https://github.com/pykale/pykale/pull/121): Update new logos
* [#125](https://github.com/pykale/pykale/pull/125): Update documentation, esp. guidance on how to use pykale

# Version  0.1.0b3

#### Code Improvements

* [#84](https://github.com/pykale/pykale/pull/84): Auto assign to the default project
* [#91](https://github.com/pykale/pykale/pull/91): MPCA pipeline
* [#93](https://github.com/pykale/pykale/pull/93): Fix black config and rerun
* [#97](https://github.com/pykale/pykale/pull/97): Add changelog CI and update logo

#### Dependencies

* [#82](https://github.com/pykale/pykale/pull/82): Remove requirements in examples and update setup

#### Tests

* [#70](https://github.com/pykale/pykale/pull/70): Add tests for utils.print
* [#80](https://github.com/pykale/pykale/pull/80): Extend automated test matrix and rename lint
* [#85](https://github.com/pykale/pykale/pull/85): Test utils logger
* [#87](https://github.com/pykale/pykale/pull/87): Test cifar/digit_access and downgrade black
* [#90](https://github.com/pykale/pykale/pull/90): Test mpca
* [#94](https://github.com/pykale/pykale/pull/94): Update test guidelines

#### Documentation Updates

* [#81](https://github.com/pykale/pykale/pull/81): Docs update version and installation
* [#88](https://github.com/pykale/pykale/pull/88): Automatically sort documented members by source order
* [#89](https://github.com/pykale/pykale/pull/89): Disable automatic docstring inheritance from parent class


# Changelog

This changelog is updated for each new release.
