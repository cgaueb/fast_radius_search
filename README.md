# Fast Radius Search Exploiting Ray Tracing Frameworks
Source Code for "Fast Radius Search Exploiting Ray Tracing Frameworks".

## Dependencies

- [tinyobjloader](https://github.com/tinyobjloader/tinyobjloader)

## Requirements

- Visual Studio 2019
- NVIDIA CUDA v10.1 (or higher)
- NVIDIA OptiX SDK v7.2.0 (or higher)

## Project Files

The main project files are:

- `main.cpp` - Invokes a set of test cases with the point clouds demonstrated in the paper
- `radius_search.hpp/cpp` - Initializes, builds and executes radius search methods using the OptiX API
- `radius_search.cu` - Implements all the relevant kernels described in the paper for radius and truncated knn searches

Additionally, the project uses some code that has been retrieved/modified from the NVIDIA OptiX framework and is located in the `sutil` folder.

## How To Build

- Add an environment variable `CUDA_PATH_V10_1` pointing to the NVIDIA CUDA installation directory, e.g. `C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.1/`. Ideally, this variable is already set during the CUDA installation
- Add an environment variable `OPTIX_PATH_V7_2` pointing to the NVIDIA OptiX installation directory, e.g. `C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.2.0/`
- Edit the `compile_optix.bat` to set the corresponding *CCBIN* path for the C++ compiler
- Edit the `compile_optix.bat` to set the *sm_arch* variable that best suites the underlying device
- Compile `assets/radius_search.cu` to `assets/ptx/radius_search.ptx` by executing the `compile_optix.bat` through the command line
- Run Visual Studio and compile the project

## How To Run

Run the `optix radius search.exe` that is located in the `x64\Release` or `x64\Debug` folder, depending on which configuration the project has been compiled with.

The expected command line output is :

- The Acceleration Data Structure (ADS) size
- The construction time of the ADS given the sample set
- The total time to execute the queries for a specific search method invocation

## How to invoke custom code

Setting up a custom routine will require to :
- Create an OptiX context
- Create an instance of `bvh_index` and initialize it
- Build the ADS based on the AABBs constructed from the samples

The available radius search functions are :
- `radius_search_count` which returns a per query counting of neighbor samples based on the given radius along with some global statistics
- `radius_search` which returns per query index and distance buffers to samples (invokes `radius_search_count` internally to guarantee the gathering of every neighbor sample)
- `truncated_knn` which returns per query index and distance buffers to samples given a user predefined maximum capacity per query

All of the above searching invocations also implement a brute force approach that simply triggers a ray generation kernel that loops over
all the available samples without invoking any BVH traversal for reference purposes.

Direct modification of `radius_search.cu` is required if a custom intersection function is required during the gathering stage.

## Future TODO

- Implement Progressive Photon Mapping as demonstrated in the paper