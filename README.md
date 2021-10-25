# relay_modules_TVM
- TVM libraries for building workloads, runtime graph modules of Convolution layers
- Provides runtime profiling for mbconv (inverted bottleneck) layers on mobile devices
    - currently supported backends: llvm(aarch64), opencl, cuda, tensorRT

- Environments:
    - TVM 0.8.dev0 or higher (needs GraphExecutor)
    - LLVM 8.0.0
