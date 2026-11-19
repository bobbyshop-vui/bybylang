## FP4-KM tile GEMM micro-kernel: every lowering against the host reference on small shapes.
##
##   CPU VM            OP_TILE_GEMM programs (dynamic and fixed rows) and the legacy shape op
##   Metal             register tiles through sessionFp4GemmMetal (simdgroup_matrix too on an
##                     Apple GPU) and a FIR-style legacy shape op (one blob in slots 2 and 3);
##                     every simdgroup_matrix source is compiled by the Metal compiler
##   Vulkan            register tiles through sessionFp4GemmTiledEnc (cooperative matrices too
##                     when the device has them); every cooperative-matrix source goes through glslc
##
## nim c -r -d:release --path:vendor/bybylang vendor/bybylang/tests/t_fp4_tile_gemm.nim

import std/[os, osproc, strutils]
import ../gpubackend
import ../micro_kernel/[session_vm, tile_gemm]
import ../backends/cpu/cpu_backend
when defined(macosx):
  import ../backends/metal/metal_backend
  import ../micro_kernel/[jit_vm, session_vm_dispatch]

proc nimf_cpu_vm_run_slots(code: ptr uint32; nWords: cint; mem: ptr float32; memLen: cint;
    packed: ptr uint8; packedLen: cint; lut: ptr float32; lutLen: cint; inF32: ptr float32;
    inLen: cint; nThreads, tgFloats, tgSize: cint; err: cstring; errLen: cint): cint {.importc, cdecl.}

type Case = object
  M, K, N: int
  x: seq[float32]
  packed: seq[uint8]     ## nibble bytes [N, K/2], then scale bytes [N, K/32]
  table: seq[float32]    ## 16 FP4 values, 256 scale values

var seed = 2463534242'u32
proc rnd(): uint32 =
  seed = seed xor (seed shl 13); seed = seed xor (seed shr 17); seed = seed xor (seed shl 5)
  seed

proc makeCase(M, K, N: int): Case =
  result = Case(M: M, K: K, N: N)
  result.x = newSeq[float32](M * K)
  for v in result.x.mitems: v = float32(int(rnd() mod 2001) - 1000) / 1000'f32
  result.packed = newSeq[uint8](N * (K div 2) + N * (K div 32))
  for b in result.packed.mitems: b = uint8(rnd() and 255)
  const fp4 = [0'f32, 0.5, 1, 1.5, 2, 3, 4, 6, -0'f32, -0.5, -1, -1.5, -2, -3, -4, -6]
  result.table = newSeq[float32](272)
  for i in 0 ..< 16: result.table[i] = fp4[i]
  for i in 0 ..< 256: result.table[16 + i] = float32(1 + (i mod 13)) / 64'f32

proc words(bytes: seq[uint8]): seq[float32] =
  result = newSeq[float32]((bytes.len + 3) div 4)
  if bytes.len > 0: copyMem(addr result[0], unsafeAddr bytes[0], bytes.len)

proc reference(c: Case): seq[float32] =
  tileGemmReference(c.x, c.packed, c.table, c.M, c.K, c.N, c.N * (c.K div 2), 0)

var failures = 0
proc check(label: string; want, got: seq[float32]) =
  var worst, peak = 0'f64
  if got.len < want.len:
    echo "FAIL ", label, ": ", got.len, " values, want ", want.len
    inc failures
    return
  for i in 0 ..< want.len:
    peak = max(peak, abs(float64(want[i])))
    let d = abs(float64(want[i]) - float64(got[i]))
    worst = if d != d: Inf else: max(worst, d)
  let ok = worst <= 1e-4 * max(1.0, peak)
  echo (if ok: "ok   " else: "FAIL "), label, " max|diff|=", formatFloat(worst, ffScientific, 2),
    " peak=", formatFloat(peak, ffDecimal, 3)
  if not ok: inc failures

proc runCpu(c: Case; program: VmProgram; packed: seq[uint8]; table: seq[float32]): seq[float32] =
  var code = program.words
  var mem = newSeq[float32](c.M * c.N)
  var bytes = packed
  var lut = table
  var x = c.x
  var err = newString(256)
  let rc = nimf_cpu_vm_run_slots(addr code[0], cint(code.len), addr mem[0], cint(mem.len),
    addr bytes[0], cint(bytes.len), addr lut[0], cint(lut.len), addr x[0], cint(x.len),
    128, 64, 128, err.cstring, 256)
  doAssert rc == 0, "CPU VM failed: " & $err.cstring
  mem

proc cpuTests(c: Case) =
  let want = reference(c)
  let tag = " M=" & $c.M & " K=" & $c.K & " N=" & $c.N
  check("cpu tile dynamic rows" & tag, want,
    runCpu(c, buildFp4TileGemmProgram(0, c.K, c.N, c.N * (c.K div 2), 0), c.packed, c.table))
  check("cpu tile fixed rows" & tag, want,
    runCpu(c, buildFp4TileGemmProgram(c.M, c.K, c.N, c.N * (c.K div 2), 0), c.packed, c.table))
  if c.K mod 64 == 0 and c.N >= 16:
    # The legacy shape op: one blob holding the nibbles, the scales and (after a word pad) the table.
    var blob = words(c.packed)
    let lutWord = fp4CoopLutWord(c.K, c.N)
    blob.setLen(lutWord)
    blob.add c.table
    var blobBytes = newSeq[uint8](blob.len * 4)
    copyMem(addr blobBytes[0], addr blob[0], blobBytes.len)
    var legacy: VmProgram
    legacy.emitFp4CoopGemm(uint32(c.M), uint32(c.K), uint32(c.N), 32, uint32(c.K div 32))
    legacy.emitHalt()
    check("cpu legacy shape op" & tag, want, runCpu(c, legacy, blobBytes, blob))
    let expanded = fp4TileProgram(legacy)
    check("cpu legacy expanded" & tag, want, runCpu(c, expanded, blobBytes, blob))

proc gpuRun(be: GpuBackend; c: Case; route: string): seq[float32] =
  doAssert sessionBegin(be), "sessionBegin"
  let hx = sessionUpload(be, c.x)
  let hw = sessionUpload(be, words(c.packed))
  let hl = sessionUpload(be, c.table)
  var o = sessionAllocScratch(be, c.M * c.N)
  let ok = case route
    of "metal-tiled": sessionFp4GemmMetal(be, hx, hw, hl, o, c.M, c.K, c.N, false)
    of "metal-simd": sessionFp4GemmMetal(be, hx, hw, hl, o, c.M, c.K, c.N, true)
    of "vulkan-tiled": sessionFp4GemmTiledEnc(be, hx, hw, hl, o, c.M, c.K, c.N)
    else: sessionFp4GemmTensorCore(be, hx, hw, hl, o, c.M, c.K, c.N)
  doAssert sessionEnd(be), "sessionEnd"
  if not ok: return @[]
  result = sessionRead(be, o, c.M * c.N)

proc compileChecks() =
  ## The Tensor Core lowerings this machine may not run: compile every row tile.
  let progs = @[buildFp4TileGemmProgram(0, 128, 96, 96 * 64, 0),
                buildFp4TileGemmProgram(20, 4096, 8192, 8192 * 2048, 0),
                (var p: VmProgram; p.emitFp4CoopGemm(37, 256, 80, 32, 8); p.emitHalt(); p)]
  let dir = getTempDir() / "t_fp4_tile_gemm"
  createDir(dir)
  for i, p in progs:
    for tile in TileGemmRowTiles:
      for frags in [1, 2]:
        let path = dir / ("coop_" & $i & "_" & $tile & "_" & $frags & ".comp")
        writeFile(path, lowerTileGemm(p, ttVulkanCoop, "k", tile, frags))
        let r = execCmdEx("glslc --target-env=vulkan1.3 -fshader-stage=compute " & quoteShell(path) & " -o " & quoteShell(path & ".spv"))
        if r.exitCode != 0:
          echo "FAIL glslc coop prog ", i, " tile ", tile, ": ", r.output; inc failures
      let tiledPath = dir / ("tiled_" & $i & "_" & $tile & ".comp")
      writeFile(tiledPath, lowerTileGemm(p, ttVulkanTiled, "k", tile))
      let r = execCmdEx("glslc --target-env=vulkan1.1 -fshader-stage=compute " & quoteShell(tiledPath) & " -o " & quoteShell(tiledPath & ".spv"))
      if r.exitCode != 0:
        echo "FAIL glslc tiled prog ", i, " tile ", tile, ": ", r.output; inc failures
      when defined(macosx):
        for t in [ttMetalSimd, ttMetalTiled]:
          let chk = metalLibraryCompileCheck(lowerTileGemm(p, t, "k_" & $t, tile))
          if not chk.ok:
            echo "FAIL Metal compile ", t, " prog ", i, " tile ", tile, ": ", chk.error; inc failures
  removeDir(dir)
  echo "compile checks done (glslc coop/tiled, Metal simdgroup/tiled)"

let cases = @[makeCase(1, 64, 16), makeCase(5, 128, 96), makeCase(20, 64, 72),
              makeCase(37, 128, 130), makeCase(70, 96, 64)]
for c in cases: cpuTests(c)
compileChecks()
when defined(macosx):
  if ensureGpuInit(gbMetal):
    for c in cases:
      let tag = " M=" & $c.M & " K=" & $c.K & " N=" & $c.N
      check("metal register tiles" & tag, reference(c), gpuRun(gbMetal, c, "metal-tiled"))
      if metalSimdgroupMatrixSupported():
        check("metal simdgroup_matrix" & tag, reference(c), gpuRun(gbMetal, c, "metal-simd"))
    # FIR micro node: the legacy shape op, the blob bound to slots 2 and 3, the lowering the FIR
    # runtime picks for this device.
    for c in cases:
      if c.K mod 64 != 0 or c.N < 16: continue
      var blob = words(c.packed)
      blob.setLen(fp4CoopLutWord(c.K, c.N))
      blob.add c.table
      var legacy: VmProgram
      legacy.emitFp4CoopGemm(uint32(c.M), uint32(c.K), uint32(c.N), 32, uint32(c.K div 32))
      legacy.emitHalt()
      let tile = q4CoopRowTile(c.M, c.N)
      let name = "t_fir_fp4_" & $c.M & "_" & $c.K & "_" & $c.N & "_r" & $tile
      let simd = metalSimdgroupMatrixSupported()
      let source = if simd: lowerFp4CoopMetal(legacy, name, tile) else: lowerFp4TiledMetal(legacy, name, tile)
      doAssert tryMetalJitCompile(source, name), "FIR-style tile GEMM compile"
      doAssert sessionBegin(gbMetal)
      let hb = sessionUpload(gbMetal, blob)
      let hx = sessionUpload(gbMetal, c.x)
      var o = sessionAllocScratch(gbMetal, c.M * c.N)
      doAssert sessionPipeline(name) and sessionBindBuffer(1, o.metalHandle) and
        sessionBindBuffer(2, hb.metalHandle) and sessionBindBuffer(3, hb.metalHandle) and
        sessionBindBuffer(4, hx.metalHandle) and
        sessionDispatch(tileGemmGroups(c.M, c.N, tile) * TileGemmThreads, TileGemmThreads)
      doAssert sessionEnd(gbMetal)
      check("metal FIR legacy shape op " & (if simd: "simdgroup_matrix" else: "register tiles") &
        " M=" & $c.M & " K=" & $c.K & " N=" & $c.N, reference(c), sessionRead(gbMetal, o, c.M * c.N))
    if not metalSimdgroupMatrixSupported():
      echo "skip metal simdgroup_matrix run: this GPU has no simdgroup_matrix (compile-checked above)"
if getEnv("T_FP4_TILE_VULKAN", "1") != "0" and ensureGpuInit(gbVulkan):
  for c in cases:
    let tag = " M=" & $c.M & " K=" & $c.K & " N=" & $c.N
    check("vulkan register tiles" & tag, reference(c), gpuRun(gbVulkan, c, "vulkan-tiled"))
    if sessionCooperativeGemmEnabled(gbVulkan) and c.K mod 64 == 0:
      check("vulkan cooperative matrix" & tag, reference(c), gpuRun(gbVulkan, c, "vulkan-coop"))
  if not sessionCooperativeGemmEnabled(gbVulkan):
    echo "skip vulkan cooperative-matrix run: no KHR_cooperative_matrix (compile-checked above)"
if failures > 0:
  quit "t_fp4_tile_gemm: " & $failures & " failure(s)"
echo "t_fp4_tile_gemm: all passed"
