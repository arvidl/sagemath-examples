# sagemath-examples

SageMath is built out of nearly [100 open-source packages](https://doc.sagemath.org/html/en/reference/spkg) and features a unified interface. SageMath can be used to study elementary and advanced, pure and applied mathematics. This includes a huge range of mathematics, including basic algebra, calculus, elementary to very advanced number theory, cryptography, numerical computation, commutative algebra, group theory, combinatorics, graph theory, exact linear algebra and much more. It combines various software packages and seamlessly integrates their functionality into a common experience. It is [well-suited](https://www.sagemath.org/library-stories.html) for education and research.
The user interface is a notebook in a web browser or the command line. Using the notebook, SageMath connects either locally to your own SageMath installation or to a SageMath server on the network. Inside the SageMath notebook you can create embedded graphics, [animations and artwork](https://wiki.sagemath.org/art),  [interactions](https://wiki.sagemath.org/interact), beautifully typeset mathematical expressions, add and delete input, and share your work across the network.

A.L @ MMIV-ML, 2022-01-28

![logo](https://www.sagemath.org/pix/logo_sagemath+icon_oldstyle.png)

See also
- **PREP Tutorials v9.4 » Introductory Sage Tutorial** at https://doc.sagemath.org/html/en/prep/Intro-Tutorial.html
- **PREP Quickstart Tutorials** at https://doc.sagemath.org/html/en/prep/quickstart.html
- **More Sage Thematic Tutorials** at https://more-sagemath-tutorials.readthedocs.io/en/latest

## Install sage 9.4 using Anaconda (not always working)
```
conda create -n sage python=3.8
conda activate sage
conda config --add channels conda-forge
conda install sage=9.4
```


## Install sage 9.4 using linux/64bit binary download  (recommended)
Download `sage-9.4-Ubuntu_20.04-x86_64.tar.bz2`from http://mirrors.mit.edu/sage/linux/64bit/index.html to your ~/SW
```
cd ~/SW
tar -xjf sage-9.4-Ubuntu_20.04-x86_64.tar.bz2

# Add the following two lines to your `.bashrc` file in your home directory:

# Sage v. 9.4
export PATH="$PATH:/home/arvid/SW/SageMath"

# To start a Jupyter Notebook instead of a Sage console, run the command:

> sage -n jupyter
```

###  Setting up SageMath as a Jupyter kernel in an existing Jupyter notebook or JupyterLab installation


```
jupyter kernelspec install --user /home/arvid/SW/SageMath/local/share/jupyter/kernels/sagemath
```

then use the generated kernel `SageMath 9.4` to access Sage functionality from a Jupyter notebook

To install a python package or library, say `pandas` and `jupyterlab` in the sage kernel:

```
> sage --pip install pandas
> sage --pip install jupyterlab
```


## Install sage 9.5rc4 Jan 24, 2022 using PyPI (might not work)
```
conda create -n sage95rc4 python=3.9
conda activate sage95rc4
pip install sagemath-standard==9.5rc4
```

## Remove `sage` environment
```
conda deactivate
conda env remove -n sage
```
