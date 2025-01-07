## Overview for GPU 

ON CPU the code is executed sequentially while on GPU it is executed in parallel. THough the CPU executes the code sequentially, to the human eye it is still to fast to see the difference. IT is a sort fo *fake parallelism*. CPUs are usually have one or multiple cores, while GPUs have thousands of cores. 

CPus have multiple cores, and next to each core is a L1 Cache, with a control Unit. The L1 Cache stores specific data for the core to use. The L1 Cache is very fast, but very small. The Control unit is a system close to the core which acts like a "scheduler" for the core. The control unit decides what the core should do next. Placing a control unit is very expensive, so this is the reason why GPUs have very few control units. CPUs have also a L2 Cache, which is bigger than the L1 Cache, but slower, as a superimposition also the L3 Cache, which is even bigger and slower.  

## Vector addition in blocks

Suppose that you have an operation of 10^6 elements. CUDA rejects an operation of 10^6 blocks it doesn't have enough cuda cores for the operation. So it diviedes the operation in blocks. Each block has a number of threads