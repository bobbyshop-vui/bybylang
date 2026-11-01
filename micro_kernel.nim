## vendor/bybylang/micro_kernel — facade package cho micro-VM + JIT.
##
## The implementation lives in `vendor/bybylang/micro_kernel/`:
##   session_vm.nim          — opcode / VmProgram / bytecode emit
##   session_vm_dispatch.nim — encode Metal/Vulkan (pipeline/bind/dispatch)
##   jit_vm.nim              — HotSpot interpret → profile → Metal/Vulkan JIT
##
## Use it through:
##   import vendor/bybylang/micro_kernel
## or (recommended at app level):
##   import nimformers   # re-exports the whole public API below
##
## GPU backend: micro_kernel imports `../gpubackend` (Metal path) and
## `../backends/vulkan/vulkan_driver` — it does not import nimformers'
## `backend.nim` (avoids a dependency cycle).

import ./micro_kernel/session_vm
import ./micro_kernel/session_vm_dispatch
import ./micro_kernel/jit_vm

export session_vm
export session_vm_dispatch
export jit_vm
