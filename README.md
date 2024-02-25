# sagemath-examples

SageMath is built out of nearly [100 open-source packages](https://doc.sagemath.org/html/en/reference/spkg) and features a unified interface. SageMath can be used to study elementary and advanced, pure and applied mathematics. This includes a huge range of mathematics, including basic algebra, calculus, elementary to very advanced number theory, cryptography, numerical computation, commutative algebra, group theory, combinatorics, graph theory, exact linear algebra and much more. It combines various software packages and seamlessly integrates their functionality into a common experience. It is [well-suited](https://www.sagemath.org/library-stories.html) for education and research.
The user interface is a notebook in a web browser or the command line. Using the notebook, SageMath connects either locally to your own SageMath installation or to a SageMath server on the network. Inside the SageMath notebook you can create embedded graphics, [animations and artwork](https://wiki.sagemath.org/art),  [interactions](https://wiki.sagemath.org/interact), beautifully typeset mathematical expressions, add and delete input, and share your work across the network.

A.L @ MMIV-ML, 2022-02-25

Using `SageMath 10.2` (see `01-sagemath-example.ipynb` for more info)

![logo](https://www.sagemath.org/pix/logo_sagemath+icon_oldstyle.png)

More info:
- **CoCalc - Collaborative Calculation and Data Science** (SageMath, Inc.) at https://cocalc.com/
- **PREP Tutorials v9.4 » Introductory Sage Tutorial** at https://doc.sagemath.org/html/en/prep/Intro-Tutorial.html
- **PREP Quickstart Tutorials** at https://doc.sagemath.org/html/en/prep/quickstart.html
- **More Sage Thematic Tutorials** at https://more-sagemath-tutorials.readthedocs.io/en/latest
- **3-manifolds** at https://github.com/3-manifolds/Sage_macOS


> See the [´01-sagemath-example.ipynb`](https://nbviewer.org/github/arvidl/sagemath-examples/blob/master/01-sagemath-example.ipynb) for illustration

<!--
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

## Install SageMath 9.0 in Ubuntu 20.04 using apt

```
sudo apt install sagemath
```

## Install SageMath 9.5 from source in Ubuntu 20.04
https://sagemanifolds.obspm.fr/install_ubuntu.html

First, install the dependencies, i.e. the Ubuntu packages required to build SageMath: simply open a terminal and type the single (long) line
```
sudo apt-get install bc binutils bzip2 ca-certificates cliquer cmake curl ecl eclib-tools fflas-ffpack flintqs g++ gengetopt git gfan gfortran glpk-utils gmp-ecm lcalc libatomic-ops-dev libboost-dev libbraiding-dev libbrial-dev libbrial-groebner-dev libbz2-dev libcdd-dev libcdd-tools libcliquer-dev libcurl4-openssl-dev libec-dev libecm-dev libffi-dev libflint-arb-dev libflint-dev libfreetype6-dev libgc-dev libgd-dev libgf2x-dev libgiac-dev libgivaro-dev libglpk-dev libgmp-dev libgsl-dev libhomfly-dev libiml-dev liblfunction-dev liblrcalc-dev liblzma-dev libm4rie-dev libmpc-dev libmpfi-dev libmpfr-dev libncurses5-dev libntl-dev libopenblas-dev libpari-dev libpcre3-dev libplanarity-dev libppl-dev libprimesieve-dev libpython3-dev libqhull-dev libreadline-dev librw-dev libsingular4-dev libsqlite3-dev libssl-dev libsuitesparse-dev libsymmetrica2-dev libz-dev libzmq3-dev libzn-poly-dev m4 make nauty openssl palp pari-doc pari-elldata pari-galdata pari-galpol pari-gp2c pari-seadata patch perl pkg-config planarity ppl-dev python3-distutils r-base-dev r-cran-lattice singular sqlite3 sympow tachyon tar tox xcas xz-utils xz-utils
```
To benefit from extra functionalities when running SageMath (e.g. exporting a Jupyter notebook to pdf), it is recommended to install some additional Ubuntu packages:
```
sudo apt-get install texlive-latex-extra texlive-xetex latexmk pandoc dvipng
```
Then you can download SageMath 9.5 sources and launch the build by typing

```
conda deactive  # If conda environment e.g. (base) is installed
```
then:
```
git clone --branch master https://github.com/sagemath/sage.git
cd sage
make configure
./configure
MAKE="make -j8" make
```
The last command launches the build in parallel on 8 threads (as specified by -j8); adapt to your CPU (usually you may choose a number of threads that is twice the number of cores of your CPU). The build time is about half an hour on a modern CPU. Once it is finished, you can run
```
./sage -n
```
A Jupyter page should then open in your browser. Click on "New" and select "SageMath 9.5" to open a Jupyter notebook with a SageMath kernel.

-->


| Notebook    |      1-Click Notebook      |
|:----------|------|
|  [01-sagemath-example.ipynb](https://nbviewer.jupyter.org/github/arvidl/sagemath-examples/blob/master/01-sagemath-example.ipynb)<br> Exploring SageMath 10.2 with Python 3.11, toolboxes, interactions and R 4.2   | [![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/arvidl/sagemath-examples/blob/master/01-sagemath-example.ipynb)|


----

## SageMath 10.2 (sage-build)

build is done in `/home/arvid/SW/Sage-10.2`  (or `/Users/arvid/SW/Sage-10.2`)
and the `sagemath` kernel will be located in 
- Ubuntu: `/home/arvid/anaconda3/envs/sage-build/share/jupyter/kernels/sagemath`
- MacOS: `/Applications/SageMath-10-2.app/Contents/Frameworks/Sage.framework/Versions/10.2/venv/share/jupyter/kernel/sagemath`

**In `~/SW/Sage-10.2` do**:

**If Ubuntu**:
```bash
git clone --branch master https://github.com/sagemath/sage.git
./bootstrap-conda
conda env create --file environment-3.11.yml --name sage-build
conda activate sage-build
./configure --with-python=$CONDA_PREFIX/bin/python --prefix=$CONDA_PREF
make -j8
```
```bash
sudo ln -sf $(pwd)/sage /usr/local/bin
```
```bash
coda deactivate   # into (base)
```

**If MacOS**:

Check: https://github.com/3-manifolds/Sage_macOS/releases <br>
and install `SageMath-10.2_x86_64.dmg` or `SageMath-10.2_arm64.dmg`

```bash
git clone --branch master https://github.com/sagemath/sage.git
./bootstrap-conda
conda env create --file environment-3.11.yml --name sage-build
conda activate sage-build
```

----

### Install `sagemath` kernel

Ubuntu:
```bash
jupyter kernelspec install --user $(sage -sh -c 'ls -d /home/arvid/anaconda3/envs/sage-build/share/jupyter/kernels/sagemath') --name sagemath-10.2 --display-name "SageMath 10.2"
```
MacOS:
```bash
jupyter kernelspec install --user $(sage -sh -c 'ls -d /Applications/SageMath-10-2.app/Contents/Frameworks/Sage.framework/Versions/10.2/venv/share/jupyter/kernel/sagemath') --name sagemath-10.2 --display-name "SageMath 10.2"
```

- Ubuntu: https://sagemanifolds.obspm.fr/install_ubuntu.html for installation (including sudo apt-get install ...)
- MacOS: https://github.com/3-manifolds/Sage_macOS/releases



