# Introduction

PyKale is a **Py**thon library for **k**nowledge-**a**ware machine **le**arning from multiple sources, particularly from multiple modalities for [multimodal learning](https://en.wikipedia.org/wiki/Multimodal_learning) and from multiple domains for [transfer learning](https://en.wikipedia.org/wiki/Transfer_learning). This library was motivated by needs in healthcare applications (hence we choose the acronym *kale*, a healthy vegetable) and aims to enable and accelerate interdisciplinary research.

## Objectives

Our objectives are to build *green* machine learning systems.

- *Reduce repetition and redundancy*:  refactor code to standardize workflow and enforce styles, and identify and remove duplicated functionalities
- *Reuse existing resources*: reuse the same machine learning pipeline for different data, and reuse existing libraries for available functionalities
- *Recycle learning models across areas*: identify commonalities between applications, and recycle models for one application to another

## API design

To achieve the above objectives, we

- design our API to be pipeline-based to unify the workflow and increase the flexibility, and
- follow core principles of standardization and minimalism in the development.

This design helps us break barriers between different areas or applications and facilitate the fusion and nurture of ideas across discipline boundaries.

## Development

We have Research Software Engineers (RSEs) on our team to help us adopt the best software engineering practices in a research context. We have modern GitHub setup with project boards, discussion, and GitHub actions/workflows. Our repository has automation and continuous integration to build documentation, do linting of our code, perform pre-commit checks (e.g. maximum file size), use [pytest](https://docs.pytest.org/en/6.2.x/) for testing and [codecov](https://about.codecov.io/) for analysis of testing.
