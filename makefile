# Makefile - BybyLang (tự detect OS: Linux / macOS / Windows)
ifeq ($(OS),Windows_NT)
    DETECTED_OS := Windows
    EXE := 
else
    DETECTED_OS := $(shell uname -s)
    EXE :=
endif

NIMC     = nim c -d:release
BIN      = bybylang$(EXE)
DEMO_SRC = demo/demo_gpu.bybylang
DEMO_OUT = demo/demo_gpu_out$(EXE)

# Cấu trúc thư mục:
#   bybylang.nim, gpubackend.nim, micro_kernel.nim -> facade + core
#   micro_kernel/                                                -> bytecode, session, JIT
#   backends/vulkan/                                             -> Vulkan backend and shaders
#   backends/metal/                                              -> Metal backend (macOS)
#   demo/                                                       -> .bybylang sample and AOT output
SRC_CORE    = bybylang.nim gpubackend.nim micro_kernel.nim
SRC_MICRO   = $(wildcard micro_kernel/*.nim)
SRC_VULKAN  = backends/vulkan/vulkan_driver.nim backends/vulkan/vulkan_caps.nim \
	backends/vulkan/q4_autotune.nim $(wildcard backends/vulkan/kernels/*.comp) \
	$(wildcard backends/vulkan/kernels/*.glsl)
SRC_METAL   = backends/metal/metal_backend.nim backends/metal/metal_shim.m backends/metal/metal_shim.h

.PHONY: build test clean

build: $(SRC_CORE) $(SRC_MICRO) $(SRC_VULKAN) $(SRC_METAL)
	@echo "[Makefile] OS: $(DETECTED_OS)"
	$(NIMC) -o:$(BIN) bybylang.nim

test: build
	./$(BIN) $(DEMO_SRC) --aot=$(DEMO_OUT)
	./$(DEMO_OUT)

clean:
	rm -f $(BIN) $(DEMO_OUT) $(DEMO_OUT).nim
	rm -rf ~/.cache/nim/$(notdir $(BIN))_r ~/.cache/nim/$(notdir $(DEMO_OUT))_r
