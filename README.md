<div align="center">

[![Garrus logo](https://github.com/sleep3r/pics/blob/main/garrus_pics/garrus-logo-big.png?raw=true)](https://github.com/sleep3r/garrus)

**In the middle of some calibrations...**

[![CodeFactor](https://www.codefactor.io/repository/github/sleep3r/garrus/badge)](https://www.codefactor.io/repository/github/sleep3r/garrus)
[![Pipi version](https://img.shields.io/pypi/v/garrus.svg)](https://pypi.org/project/garrus/)
[![Docs](https://img.shields.io/badge/dynamic/json.svg?label=docs&url=https%3A%2F%2Fpypi.org%2Fpypi%2Fgarrus%2Fjson&query=%24.info.version&colorB=brightgreen&prefix=v)](https://github.com/sleep3r/garrus/wiki)
[![PyPI Status](https://pepy.tech/badge/garrus)](https://pepy.tech/project/garrus)

[![Telegram](https://img.shields.io/badge/author-telegram-blue)](https://t.me/sleep3r)
[![Github contributors](https://img.shields.io/github/contributors/sleep3r/garrus.svg?logo=github&logoColor=white)](https://github.com/sleep3r/garrus/graphs/contributors)

[![python](https://img.shields.io/badge/python_3.6-passing-success)](https://github.com/sleep3r/garrus/badge.svg?branch=main&event=push)
[![python](https://img.shields.io/badge/python_3.7-passing-success)](https://github.com/sleep3r/garrus/badge.svg?branch=main&event=push)
[![python](https://img.shields.io/badge/python_3.8-passing-success)](https://github.com/sleep3r/garrus/badge.svg?branch=main&event=push)

</div>

Garrus is a python framework for better confidence estimate of deep neural networks. Modern networks are overconfident estimators, that makes themselves unreliable and therefore limits the deployment of them in safety-critical applications.

Garrus provides tools for high quality confidence estimation such as confidence calibration and ordinal ranking methods, helping networks to **know correctly what they do not know**. 

----

## Installation:
```bash
pip install -U garrus
```

## Documentation:
  - [master v0.2.0](https://github.com/sleep3r/garrus/wiki/0.2.0@master-documentation)
  - [develop v0.2.0](https://github.com/sleep3r/garrus/wiki/0.2.0@develop-documentation)

## Roadmap:
- Core:
  - Calibration metrics:
    - [x] ECE
    - [x] NLL
    - [x] Brier
  - Ordinal Ranking Metrics:
    - [x] AURC
    - [x] E-AURC
    - [x] AUPRE
    - [x] FPR-n%-TPR
  - Visualizations:
    - [x] Reliability Diagram
    - [x] Confidence Histogram
  - [ ] Garrus Profiling
- Confidence Calibration:
    - Scaling:
      - [x] Platt
      - [x] Temperature
    - Binning: 
      - [ ] Histogram
      - [ ] Isotonic Regression
- Confidence Regularization:
  - Losses:
    - [ ] Correctness Ranking Loss
    - [ ] Focal Entropy Penalized Loss
  - [ ] Language Model Beam Search
- Confidence Networks:
  - [ ] ConfidNet
  - [ ] GarrusNet

---

### Citation:
Please use this bibtex if you want to cite this repository in your publications:

    @misc{garrus,
        author = {Kalashnikov, Alexander},
        title = {Deep neural networks calibration framework},
        year = {2021},
        publisher = {GitHub},
        journal = {GitHub repository},
        howpublished = {\url{https://github.com/sleep3r/garrus}},
    }
 
### References:
|Papers|
|---|
| [[1]](https://arxiv.org/pdf/1706.04599.pdf) Guo, Chuan, et al. "On calibration of modern neural networks." International Conference on Machine Learning. PMLR, 2017. APA |
| [[2]](https://arxiv.org/pdf/2007.01458.pdf) Moon, Jooyoung, et al. "Confidence-aware learning for deep neural networks." international conference on machine learning. PMLR, 2020. |
| [[3]](https://arxiv.org/pdf/1909.10155.pdf) Kumar, Ananya, Percy Liang, and Tengyu Ma. "Verified uncertainty calibration." arXiv preprint arXiv:1909.10155 (2019). |