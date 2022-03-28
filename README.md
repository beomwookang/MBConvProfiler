# MBConvProfiler
- TVM libraries for building workloads, runtime graph modules of Convolution layers
- Provides runtime profiling for mbconv (inverted bottleneck) layers on mobile devices
    - currently supported backends: llvm(aarch64), opencl, cuda, tensorRT

### Environments:
- TVM 0.8.dev0 or higher (needs GraphExecutor)
- LLVM 8.0.0
- OpenCL (aarch64)

### Usage Example
```
#TVM

from mbconv_profiler import MBProfiler

mbprof = MBProfiler('firefly', 'cpu', opt_level=3)
mbprof.do_exec((112,3,2,16,96,24))      #(input_size, kernel_size, stride, input_channel, expanded_channel, output_channel)

print(mbprof.exec_record)
```
