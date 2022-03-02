# rsHRF (Resting State HRF Estimation and Deconvolution)

## HemodynamicResponseModeling

(From: Takuya Ito's https://github.com/ito-takuya/HemodynamicResponseModeling
modeling hemodynamic response functions of the BOLD signal using Windkessel-Balloon model)

Based on Friston KJ, Harrison L, Penny W. Dynamic causal modelling. Neuroimage 2003;19:1273–1302.


### Hemodynamic state equations

The remaining state variables of each region are biophysical states engendering the BOLD signal and mediate the
translation of neuronal activity into hemodynamic responses. Hemodynamic states are a function of, and only of,
the neuronal state of each region. These equations have been described elsewhere (Friston et al., 2000) and constitute a
hemodynamic model that embeds the _Balloon–Windkessel model_ (Buxton et al., 1998; Mandeville et al., 1999).

![img](assets/hemodynamic_model.png)



#### Takuya Ito implementation of Friston KJ et al. Dynamic causal modelling. Neuroimage 2003;19:1273-1302.
```
def balloonWindkessel(z, sampling_rate, alpha=0.32, kappa=0.65, gamma=0.41, tau=0.98, rho=0.34, V0=0.02):
    """
    Computes the Balloon-Windkessel transformed BOLD signal
    Numerical method (for integration): Runge-Kutta 2nd order method (RK2)

    z:          Measure of neuronal activity (space x time 2d array, or 1d time array)
    sampling_rate: sampling rate, or time step (in seconds)
    alpha:      Grubb's exponent
    kappa:      Rate of signal decay (in seconds)
    gamma:      Rate of flow-dependent estimation (in seconds)
    tau:        Hemodynamic transit time (in seconds)
    rho:        Resting oxygen extraction fraction
    V0:         resting blood vlume fraction

    RETURNS:
    BOLD:       The transformed BOLD signal (from neural/synaptic activity)
    s:          Vasodilatory signal
    f:          blood inflow
    v:          blood volume
    q:          deoxyhemoglobin content
    """
```


## Nilearn: Example of hemodynamic response functions.

The HRF is the filter that couples neural responses to the metabolic-related changes in the MRI signal. HRF models are simply phenomenological.
- https://nilearn.github.io/auto_examples/04_glm_first_level/plot_hrf.html


## rsHRF: A toolbox for resting-state HRF estimation and deconvolution

- Guo-Rong Wu, Nigel Colenbier, Sofie Van Den Bossche, Kenzo Clauw, Amogh Johri, Madhur Tandon, Daniele Marinazzo. “rsHRF: A Toolbox for Resting-State HRF Estimation and Deconvolution.” Neuroimage, 2021, 244: 118591. [[link](https://www.sciencedirect.com/science/article/pii/S1053811921008648)]


- http://bids-apps.neuroimaging.io/rsHRF
>Resting state HRF estimation from BOLD-fMRI signal. This toolbox is aimed to retrieve the onsets of pseudo-events triggering an hemodynamic response from resting state fMRI BOLD voxel-wise signal. It is based on [point process](https://en.wikipedia.org/wiki/Point_process) theory, and fits a model to retrieve the optimal lag between the events and the HRF onset, as well as the HRF shape, using either the canonical shape with two derivatives, or a (smoothed) Finite Impulse Response.
>
>Once that the HRF has been retrieved for each voxel, it can be deconvolved from the time series (for example to improve lag-based connectivity estimates), or one can map the shape parameters everywhere in the brain (including white matter), and use the shape as a pathophysiological indicator.


![img](https://raw.githubusercontent.com/guorongwu/rsHRF/master/docs/BOLD_HRF.png)




### rsHRF Installation and Setup
This App can be used as a standalone Python Package OR as a BIDS-App through Docker.

### Standalone Python Package (Command Line Interface)
To be used as a command line tool, ensure that you have Python>=3.6 and use the command pip3 install rsHRF. This command takes care of all the necessary dependencies so that the tool is usable straight out of the box. Once done, run rsHRF --help to see the required positional and optional arguments. The command line for the app installed in this way is rsHRF.

**We will install and use `rsHRF` in the [`SageMath`](https://www.sagemath.org) `9.5` environment:**
```
conda deactivate                                             # no conda environments (even not "base")
sage --pip install rsHRF                                     # install the rsHRF toolbox in sage
sage --pip install nilearn                                   # install the Nilearn package in sage
sage --pip install mat4py                                    # load & save data in Matlab format in sage
sage --pip install git+https://github.com/ANTsX/ANTsPy.git   # install ANTsPy package in sage (long time)

sage -n                                          # start a Jupyter Notebook (with the SageMath 9.5 kernel)
```

### Usage

http://bids-apps.neuroimaging.io/rsHRF/

#### The input:

There are 3 ways one can input data to this application.

- A standalone .nii / .nii.gz or .gii / .gii.gz file. This option can be accessed using the --input_file optional argument followed by the path of the file.

- A standalone .txt file. This option can be accessed using the --ts optional argument followed by the path of the file.

- A BIDS formatted data-set directory. This option can be accessed using the --bids_dir optional argument followed by the path of the directory. This requires the input dataset to be in valid BIDS format, and have a derivatives type with preprocessed resting-state fMRI data. We highly recommend that you validate your dataset with the free, online BIDS Validator.

Out of the above 3 options, one of them is always required and more than one cannot be supplied at once.

#### The mask / atlas files:

Mask files are only provided along with the --input_file or bids_dir argument. There are 2 ways one can supply the corresponding mask / atlas files.

- A standalone .nii / .nii.gz or .gii / .gii.gz file. This option can be accessed using the --atlas optional argument followed by the name of the file.

- The --brainmask argument which tells the application that the mask files are present within the BIDS formatted data-set directory itself (which was supplied with --bids_dir).

Out of the above 2 options, both cannot be supplied at once. In the case where neither of the 2 options are supplied, the app proceeds to generate a mask by computing the variance, however, providing a mask-file is strongly recommended.

Also, --input_file and --brainmask together are an invalid combination.

The 5 use-cases are explained below:

- --input_file : The standalone .nii / .nii.gz or .gii / .gii.gz file is passed to the application for the analysis, the mask gets generated by computing the variance, and the outputs are determined accordingly.

- --ts : The standalone .txt file containing a time-series (floating point values separated by line breaks), or multiple time-series’ (, separated floating point values, where one time-series corresponds to a column). No mask file is required in this scenario, and the outputs are determined accordingly.

- --input_file and --atlas : The standalone atlas and the standalone input_file are passed to the application for the analysis and the outputs are determined accordingly.

- --bids_dir and --atlas : The standalone atlas is used with ALL the input files present in the bids_dir directory. Thus, the atlas serves as a common mask for the whole BIDS formatted data-set.

- --bids_dir and --brainmask : This should be used when for each input file present in the BIDS formatted data-set, the corresponding mask file exists within the same data-set. The application then pairs the input_files with their corresponding masks provided that the 2 files share a common prefix.


#### The output directory:

The output directory is accessed using --output_dir, and is the folder under which all the resulting .nii / .gii / .mat / .png files will be stored. The application further makes folders for each of the participants / subjects if the input is supplied through the argument --bids_dir. In the case of --ts argument, all the output types are stored in the .mat file. It is mandatory to provide an output directory.


#### The Analysis Level:

There are 2 kinds of analysis that can be performed. This can be accessed using --analysis_level.

- participant : participant level analysis performs the analysis for each (or some) subject(s) present in the BIDS formatted data-set. This is mandatory when the input is supplied with --bids_dir as group level analysis is not supported yet. This should not be supplied when input is supplied with --input_file argument. Doing so shall result in an error.

- group : Coming Soon! - Use participant for now.


#### The Analysis Method:

The analysis can be carried out using 6 estimation methods.

These are canon2dd, gamma, fourier, hanning, sFIR and FIR.

One of them needs to be supplied using the --estimation argument followed by one of the above 3 choices.


#### The input parameters:

There are many input parameters that can be supplied to customize the analysis. Please see all of them under the Parameters heading under the documentation by running `rsHRF --help`.


### Example Use-Cases

From: http://bids-apps.neuroimaging.io/rsHRF/
#### Running the analysis with a single input file (.txt)
a) Through the Python Package

`rsHRF --ts input_file.txt --estimation hanning --output_dir results -TR 2.0`

In the above example, the input file is a .txt file input_file.txt. The estimation method passed is hanning. The -TR argument (which represents the BOLD repetition time) needs to be supplied here.

#### Running the analysis with a single input file (.nii / .nii.gz or .gii / .gii.gz)
a) Through the Python Package

`rsHRF --input_file input_file --estimation fourier --output_dir results`

In the above example, the input_file can be a .nii/.nii.gz or .gii/ gii.gz image. The estimation method passed is fourier. The -TR argument (which represents the BOLD repetition time) needs to be supplied if a .gii/.gii.gz input file is used.

#### Running the analysis with a single input file and a single mask file.
a) Through the Python Package

`rsHRF --input_file input_file --atlas mask_file --estimation canon2dd --output_dir results`

In the above example, the `input_file` can be a .nii/.nii.gz or .gii/ gii.gz image. The `output_dir` is the results directory. The corresponding mask is the mask_file that should have a matching extension (.nii/.nii.gz or .gii/gii.gz) with the input_file. The estimation method passed is canon2dd. The analysis level is not to be supplied here. The -TR argument (which represents the BOLD repetition time) needs to be supplied if .gii / .gii.gz input file is used.

#### Running the analysis with a BIDS formatted data-set that also includes a unique mask file for each of the input file present.

Note: By default all input files in the BIDs directory need to have the suffix of the type *_bold.nii or *_bold.nii.gz. The corresponding mask files in the BIDs directory need to be of the type *_mask.nii or *_mask.nii.gz. Also, they must be present in the func directory under their respective subject / session folder. Furthermore, two corresponding input and mask files need to have the same prefix.

For example, 2 corresponding input and mask files according to BIDS format can be input_bold.nii and input_mask.nii. These 2 will then be paired up for analysis.

a) Through the Python Package

`rsHRF --bids_dir input_dir --output_dir results --analysis_level participant --brainmask --estimation canon2dd --participant_label 001 002`

In the above example, the output directory is results directory. The BIDS formatted data-set lies in the input_dir directory. The associated mask files also lie within the BIDS dataset. The analysis level is participant. The analysis is performed only for sub-001 & sub-002 out of all the available subjects in the BIDS dataset.


#### Running the analysis using BIDS-filter to input certain files to BIDS-App.
a) Through the Python Package

`rsHRF --bids_dir input_dir --output_dir results --analysis_level participant --bids_filter_file bids_filter_file.json --brainmask --estimation canon2dd --participant_label 001 002`

In the above example, the output directory is results directory. The BIDS formatted data-set lies in the input_dir directory. The associated mask files also lie within the BIDS dataset. The analysis level is participant. A custom bids_filter_file.json to filter the BIDS-data input to the BIDS-App. The analysis is performed only for sub-001 & sub-002 out of all the available subjects in the BIDS dataset.
