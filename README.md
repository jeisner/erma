Overview
========

ERMA is a software package that implements general CRFs (and MRFs) with no restriction to the model structure. The package includes a training algorithm that performs Empirical Risk Minimization under Approximations (ERMA) as described  here and here). ERMA can optimize several loss functions such as Accuracy, MSE and F-score. ERMA is available freely for academic use. 

Features
========

* Unrestricted model structure.
* Loss-aware and approximation-aware minimum-risk training via Empirical Risk Minimization under Approximations (ERMA).
* A syntax for specifying model structure.
* Implemented in Java: easy to port to different architectures.
* Generic and easily extensible.

Citations
=========

If you use ERMA for research purposes, please cite:

   @inproceedings{stoyanov2011,
     title={Empirical Risk Minimization of Graphical Model Parameters Given Approximate Inference, Decoding, and Model Structure},
     author={Veselin Stoyanov, and Alexander Ropson, and Jason Eisner.},
     booktitle={Proceedings of the Fourteenth International Conference on Artificial Intelligence and Statistics},
     volume={15},
     year={2011}
   }

or 

   @inproceedings{stoyanov2012,
     title={Minimum-Risk Training of Approximate {CRF}-Based {NLP} Systems},
     author={Veselin Stoyanov and Jason Eisner},
     booktitle={Proceedings of NAACL},
     year={2012}
   }

