# Third-Party Notices for PyKale

PyKale is distributed under its repository-level MIT License. Portions of the
source tree are adapted, modified, refactored, or otherwise derived from the
third-party projects listed below. Those portions remain subject to the
applicable upstream licence terms and notices.

This notice covers **source code incorporated into PyKale**, rather than every
package that PyKale imports or installs as an ordinary dependency. Ordinary
runtime/development dependencies should continue to be managed through the
package metadata and their own distributions.

Upstream licence files are reproduced under [`licenses/`](licenses/). The files
in this bundle were prepared from the current upstream repositories reviewed on
2026-10-06. If PyKale later pins provenance to a specific upstream commit, the
corresponding licence/notice from that commit should be retained.

## Incorporated or adapted third-party source

| Project | Upstream source | Licence | Local licence file | PyKale use / provenance |
|---|---|---|---|---|
| pytorch-ada | https://github.com/criteo-research/pytorch-ada | Apache-2.0 | `licenses/pytorch-ada-LICENSE.txt` | Domain-adaptation, data-loading, sampling and related model utilities adapted from upstream source. |
| MultiBench | https://github.com/pliang279/MultiBench | MIT | `licenses/MultiBench-LICENSE.txt` | Multimodal fusion, AVMNIST loading and neural-network training code adapted/refactored from upstream source. |
| ISONet | https://github.com/HaozhiQi/ISONet | MIT | `licenses/ISONet-LICENSE.txt` | ISONet model and related utilities based on/adapted from upstream implementation. |
| MOGONET | https://github.com/txWang/MOGONET | MIT | `licenses/MOGONET-LICENSE.txt` | MOGONET implementation refactored into PyKale/PyG components. |
| DrugBAN | https://github.com/peizhenbai/DrugBAN | MIT | `licenses/DrugBAN-LICENSE.txt` | Drug–target interaction data/model/training functionality refactored for PyKale. |
| BAN-VQA | https://github.com/jnhwkim/ban-vqa | MIT | `licenses/BAN-VQA-LICENSE.txt` | `FCNet` implementation adapted from BAN-VQA source. |
| tsn-pytorch | https://github.com/yjxiong/tsn-pytorch | BSD-2-Clause | `licenses/tsn-pytorch-LICENSE.txt` | Video dataset loading/sampling functionality adapted from Temporal Segment Networks source. |
| scikit-learn | https://github.com/scikit-learn/scikit-learn | BSD-3-Clause | `licenses/scikit-learn-LICENSE.txt` | Cross-validation/model-selection source modified for PyKale functionality. |
| pytorch-i3d | https://github.com/piergiaj/pytorch-i3d | Apache-2.0 | `licenses/pytorch-i3d-LICENSE.txt` | I3D implementation used as a source for PyKale's I3D module. |
| kinetics-i3d | https://github.com/google-deepmind/kinetics-i3d | Apache-2.0 | `licenses/kinetics-i3d-LICENSE.txt` | I3D reference implementation used as a source for PyKale's I3D module. |
| PyTorch | https://github.com/pytorch/pytorch | BSD-style | `licenses/PyTorch-LICENSE.txt` | Selected reference/tutorial source has been copied or modified in PyKale modules. |
| Torchvision | https://github.com/pytorch/vision | BSD-3-Clause | `licenses/Torchvision-LICENSE.txt` | Video ResNet/reference source modified for PyKale. |

## Licence handling notes

- The repository-level PyKale `LICENSE` can remain MIT.
- Copyright and licence notices attached to incorporated third-party source
  should be retained with redistributed source/substantial portions.
- For Apache-2.0-derived files, modified files should retain applicable
  attribution and carry an indication that changes were made.
- If an upstream Apache-2.0 project supplies a `NOTICE` file relevant to the
  incorporated source, the applicable NOTICE content should also be retained.
- This file should be updated when new third-party source is copied, adapted,
  or refactored into PyKale.
