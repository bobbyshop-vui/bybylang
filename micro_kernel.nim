## vendor/bybylang/micro_kernel — facade package cho micro-VM + JIT.
##
## Nội dung thật nằm trong `vendor/bybylang/micro_kernel/`:
##   session_vm.nim          — opcode / VmProgram / bytecode emit
##   session_vm_dispatch.nim — encode Metal/Vulkan (pipeline/bind/dispatch)
##   jit_vm.nim              — HotSpot interpret → profile → Metal/Vulkan JIT
##
## Dùng qua:
##   import vendor/bybylang/micro_kernel
## hoặc (khuyến nghị app-level):
##   import nimformers   # đã re-export toàn bộ API public dưới đây
##
## Backend GPU: micro_kernel import `../gpubackend` (Metal path) và
## `../backends/vulkan/vulkan_driver` — không import `backend.nim` của
## nimformers (tránh vòng phụ thuộc).

import ./micro_kernel/session_vm
import ./micro_kernel/session_vm_dispatch
import ./micro_kernel/jit_vm

export session_vm
export session_vm_dispatch
export jit_vm
