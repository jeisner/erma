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

To build a single jar with all the dependencies included:

    mvn compile assembly:single


TUTORIAL
========

To follow the tutorial example from the ERMA website
(https://sites.google.com/site/ermasoftware/getting-started), run the following commands:
    
    # Create a new directory with all the appropriate files.
    mkdir tmp
    export TUTORIAL=src/main/resources/tutorial
    export CONFIG=src/main/config/toy.cfg
    cp -r src/main/resources/tutorial tmp/tutorial
    cp -r src/main/config tmp/config
    # Run the commands within that directory using the jar in the target directory.
    cd tmp    
    python ../src/main/python/txt2ff.py tutorial/toy.train.txt
    java -cp ../target/erma-1.0.1-SNAPSHOT-jar-with-deps.jar driver.Learner -config=config/toy.cfg
    java -cp ../target/erma-1.0.1-SNAPSHOT-jar-with-deps.jar driver.Tester -config=config/toy.cfg -data=tutorial/toy.test.data.ff -features=tutorial/toy.trained-best.ff
    java -cp ../target/erma-1.0.1-SNAPSHOT-jar-with-deps.jar driver.Classifier -config=config/toy.cfg -data=tutorial/toy.test.data.ff -features=tutorial/toy.trained-best.ff -pred_fname=tutorial/toy.test.data.predictions
