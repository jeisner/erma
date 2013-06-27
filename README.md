Overview
========

ERMA is a software package that implements general CRFs (and MRFs)
with no restriction to the model structure. The package includes a
training algorithm that performs Empirical Risk Minimization under
Approximations (ERMA) as described 
[here](http://cs.jhu.edu/~jason/papers/#stoyanov-ropson-eisner-2011)
and [here](http://cs.jhu.edu/~jason/papers/#stoyanov-eisner-2012-naacl).
ERMA can optimize several loss functions such as Accuracy, MSE and
F-score.  

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

    @InProceedings{stoyanov-ropson-eisner-2011,
      author =      {Veselin Stoyanov and Alexander Ropson and Jason
		     Eisner},
      title =       {Empirical Risk Minimization of Graphical Model
		     Parameters Given Approximate Inference, Decoding, and
		     Model Structure},
      booktitle =   {Proceedings of the 14th International Conference on
		     Artificial Intelligence and Statistics (AISTATS)},
      series =      {JMLR Workshop and Conference Proceedings},
      volume =      {15},
      pages =       {725--733},
      note =        {Supplementary material (4 pages) also available},
      year =        {2011},
      month =       apr,
      address =     {Fort Lauderdale},
      url =         {http://cs.jhu.edu/~jason/papers/#stoyanov-ropson-eisner-2011}
    }

or 

    @InProceedings{stoyanov-eisner-2012-naacl,
      author =      {Veselin Stoyanov and Jason Eisner},
      title =       {Minimum-Risk Training of Approximate {CRF}-Based {NLP}
		     Systems},
      booktitle =   {Proceedings of NAACL-HLT},
      pages =       {120--130},
      year =        {2012},
      month =       jun,
      address =     {Montreal},
      url =         {http://cs.jhu.edu/~jason/papers/#stoyanov-eisner-2012-naacl}
    }

BUILD
========

To build the project:

    mvn compile

To create Eclipse project files:

    mvn eclipse:eclipse

The above command will add the .project and .classpath files with the
M2_REPO classpath variable. In Eclipse open the Preferences and
navigate to 'Java --> Build Path --> Classpath Variables'. Add a new
classpath variable M2_REPO with the path to your local repository
(e.g. ~/.m2/repository).
