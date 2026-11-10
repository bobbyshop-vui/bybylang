## CPU JIT for micro-VM bytecode.
##
## The Metal lowering of `jit_vm` already turns a bytecode program into straight-line C-like code
## (registers become locals, FOR/IF become real loops). This module wraps that body in a plain C++
## function, compiles every kernel of a model in one shared library with the system compiler, and
## loads it with dlopen, so the CPU runs compiled code instead of interpreting the bytecode.
## Programs that need threadgroup memory, barriers, atomics, grid control or the input domain are
## reported as unsupported and keep using the interpreter in `cpu_vm.c`.
##
## Environment:
##   BYBY_CPU_JIT=0        disable (always interpret)
##   BYBY_CPU_JIT_DIR=...  cache directory (default ~/Library/Caches/nimformers/cpu-jit)
##   CXX=...               compiler (default c++)

import std/[os, osproc, strutils, tables, dynlib]
import ../../micro_kernel/session_vm
import ../../micro_kernel/jit_vm

type
  CpuJitUnit* = tuple[name: string, source: string]

const unsupportedMarkers = ["threadgroup_barrier", "atomic", "ldShared", "float4", "tgF32",
  "memF32_", "gLid", "ctrl_", "unknown op", "inF32", "vact_"]

const prelude = """
#include <math.h>
#include <stdint.h>
#include <string.h>
typedef uint32_t uint;
typedef uint8_t uchar;
#define device
#define threadgroup
template <typename T, typename S> static inline T as_type(S s) { T t; memcpy(&t, &s, sizeof(T)); return t; }
static inline float rsqrt(float x) { return 1.0f / sqrtf(x); }
"""

proc cpuJitEnabled*(): bool =
  getEnv("BYBY_CPU_JIT", "1") != "0"

proc fnv64(text: string): uint64 =
  result = 14695981039346656037'u64
  for ch in text:
    result = (result xor uint64(ord(ch))) * 1099511628211'u64

proc cpuJitFunctionName*(p: VmProgram): string =
  var text = newStringOfCap(p.words.len*9)
  for w in p.words:
    text.add toHex(cast[int64](w), 8)
  "fjit_" & toHex(cast[int64](fnv64(text)), 16)

proc cpuJitSource*(p: VmProgram; fnName: string): string =
  ## C++ source of one kernel, or "" when the interpreter must run it.
  let metal = lowerBytecodeToMetal(p, fnName)
  let head = "[[threadgroup_position_in_grid]])\n{\n"
  let start = metal.find(head)
  let stop = metal.rfind("}")
  if start < 0 or stop <= start:
    return ""
  let body = metal[start+head.len ..< stop]
  for marker in unsupportedMarkers:
    if marker in body:
      return ""
  prelude & "\nstatic inline void " & fnName & "_t(float* memF32, const uchar* memU8, const float* lut, uint tid, uint lid, uint tgid)\n{\n" & body & "\n}\n" &
    "extern \"C\" void " & fnName & "(float* mem, int n, int tgs)\n{\n" &
    "  for (int t = 0; t < n; ++t) " & fnName & "_t(mem, (const uchar*)mem, mem, (uint)t, (uint)(t % tgs), (uint)(t / tgs));\n}\n"

proc cacheDir(): string =
  result = getEnv("BYBY_CPU_JIT_DIR")
  if result.len == 0:
    result = getHomeDir() / (when defined(macosx): "Library/Caches/nimformers/cpu-jit" else: ".cache/nimformers/cpu-jit")

var cpuJitLastError* = ""

proc cpuJitBuild*(units: seq[CpuJitUnit]): Table[string, pointer] =
  ## Compile (or reuse from the cache) one library holding all `units`; map function name -> address.
  result = initTable[string, pointer]()
  if units.len == 0 or not cpuJitEnabled():
    return
  var combined = prelude
  var names: seq[string]
  for unit in units:
    # Each unit already carries the prelude; keep one copy only.
    combined.add unit.source[prelude.len .. ^1]
    names.add unit.name
  let dir = cacheDir()
  createDir(dir)
  let stem = toHex(cast[int64](fnv64(combined)), 16)
  let lib = dir / ("k_" & stem & (when defined(macosx): ".dylib" else: ".so"))
  if not fileExists(lib):
    let src = dir / ("k_" & stem & ".cpp")
    let tmp = lib & "." & $getCurrentProcessId() & ".tmp"
    writeFile(src, combined)
    let cxx = getEnv("CXX", "c++")
    let cmd = cxx & " -O3 -std=c++17 -ffp-contract=off -shared -fPIC -w -o " & quoteShell(tmp) & " " & quoteShell(src)
    let (output, code) = execCmdEx(cmd)
    if code != 0:
      cpuJitLastError = output
      return
    moveFile(tmp, lib)
  let handle = loadLib(lib)
  if handle == nil:
    cpuJitLastError = "cannot load " & lib
    return
  for name in names:
    let address = symAddr(handle, name)
    if address != nil:
      result[name] = address

type CpuJitFn = proc(mem: ptr float32; n, tgs: cint) {.cdecl.}

proc cpuJitRun*(fn: pointer; mem: var seq[float32]; nThreads, tgSize: int): bool =
  if fn == nil or mem.len == 0 or nThreads <= 0 or tgSize <= 0:
    return false
  cast[CpuJitFn](fn)(addr mem[0], cint(nThreads), cint(tgSize))
  true
