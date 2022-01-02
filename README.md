# sagemath-examples

SageMath is built out of nearly [100 open-source packages](https://doc.sagemath.org/html/en/reference/spkg) and features a unified interface. SageMath can be used to study elementary and advanced, pure and applied mathematics. This includes a huge range of mathematics, including basic algebra, calculus, elementary to very advanced number theory, cryptography, numerical computation, commutative algebra, group theory, combinatorics, graph theory, exact linear algebra and much more. It combines various software packages and seamlessly integrates their functionality into a common experience. It is [well-suited](https://www.sagemath.org/library-stories.html) for education and research.
The user interface is a notebook in a web browser or the command line. Using the notebook, SageMath connects either locally to your own SageMath installation or to a SageMath server on the network. Inside the SageMath notebook you can create embedded graphics, [animations and attwork](https://wiki.sagemath.org/art),  beautifully typeset mathematical expressions, add and delete input, and share your work across the network.

A.L @ MMIV-ML, 2022-01-02

![logo](https://www.sagemath.org/pix/logo_sagemath+icon_oldstyle.png)

See also 
- **PREP Tutorials v9.4 » Introductory Sage Tutorial** at https://doc.sagemath.org/html/en/prep/Intro-Tutorial.html 
- **PREP Quickstart Tutorials** at https://doc.sagemath.org/html/en/prep/quickstart.html 
- **More Sage Thematic Tutorials** at https://more-sagemath-tutorials.readthedocs.io/en/latest

## Install sage using Anaconda
```
conda create -n sage python=3.8
conda activate sage
conda config --add channels conda-forge
conda install sage=9.4
```

## Remove `sage` environment
```
conda deactivate
conda env remove -n sage
```
