## README - A-KAZE Features
This is a fork of libAKAZE with modifications to run it on the GPU using CUDA. The working branch is `cpuidentity`

The interface is the same as the original version. Just changing namespace from libAKAZE to libAKAZECU should be enough. Keypoins and descriptors are returned on the CPU for later matching etc. using e.g. OpenCV. We also provide a rudimentary brute force matcher running on the GPU.

For a detailed description, please refer to <https://github.com/pablofdezalc/akaze>

This code was created as a joint effort between

- Mårten Björkman <https://github.com/Celebrandil>
- Alessandro Pieropan <https://github.com/CoffeRobot>


## Optimizations
The code has been optimized with the goal to maintain ths same interface as well as to produce the same results as the original code. This means that certain tradeoffs have been necessary, in particular in the way keypoints are filtered. One difference remain though related to finding scale space extrema, which has been reported as an issue here:

<https://github.com/pablofdezalc/akaze/issues/24>

Major optimizations are possible, but this is work in progress. These optimizations will relax the constraint to have identical results as the original code.



## Benchmarks
The following benchmarks are measured on the img1.pgm in the iguazu dataset provided by the original authors, and are averages over 100 runs. The computer is a 16 core Xeon running at 2.6 GHz with 32 GB of RAM and an Nvidia Titan X (Maxwell). The operating system is Ubuntu 14.04, with CUDA 8.0.

| Operation     | CPU (original) (ms)      | GPU (ms)  |
| ------------- |:------------------------:|:---------:|
| Detection     |            117           |    6.5    |
| Descriptor    |            10            |    0.9    |

## Limitations
- The CUDA version currently comes with some limitations. The maximum number of detected keypoints _before_ filtering is 8192, and the maximum number per level is 2048. This constraint might be relatex in future updates. 
- The only descriptor available is MLDB, as proposed in the original authors' paper.
- Currently it only works with 4 octaves and 4 sub-levels (default settings).

## Citation
If you use this code as part of your research, please cite the following papers:

CUDA version

1. **Feature Descriptors for Tracking by Detection: a Benchmark**. Alessandro Pieropan, Mårten Björkman, Niklas Bergström and Danica Kragic (arXiv:1607.06178).

Original A-KAZE papers

2. **Fast Explicit Diffusion for Accelerated Features in Nonlinear Scale Spaces**. Pablo F. Alcantarilla, J. Nuevo and Adrien Bartoli. _In British Machine Vision Conference (BMVC), Bristol, UK, September 2013_

3. **KAZE Features**. Pablo F. Alcantarilla, Adrien Bartoli and Andrew J. Davison. _In European Conference on Computer Vision (ECCV), Fiorenze, Italy, October 2012_


## CPU Implementations
If the GPU implementation isn't an option for you, have a look at the CPU-versions below

- <https://github.com/pablofdezalc/akaze>, the original code
- <https://github.com/h2suzuki/fast_akaze>, a faster implementation of the original code


## MATLAB interface

This will presumably not work, but might only require a few modifications. If someone is interested, fork the repository and create a pull-request.


## Contact Info
If you have questions, or are finding this code useful, please let me know!

Niklas Bergström,
email: nbergst@gmail.com
