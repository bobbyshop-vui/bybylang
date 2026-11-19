## tile_gemm.nim — lowering of the tile GEMM micro-kernel (OP_TILE_GEMM programs).
##
## A tile GEMM program (session_vm.buildFp4TileGemmProgram / buildF16TileGemmProgram) computes
## out[M, N] = x[M, K] * W^T with FP4-KM (format 0) or FP16 (format 1, Vulkan targets) weights.
## Its ops name the steps of one threadgroup's work - clear the accumulator, per K slice stage the
## activations and the weights, multiply-accumulate, then store - and this module turns every op
## into source for the device it will run on:
##
##   ttVulkanCoop   GLSL, KHR_cooperative_matrix: FP16 16x16 fragments, FP32 accumulation
##   ttMetalSimd    MSL, simdgroup_matrix: FP32 8x8 fragments (Apple GPU family 7+)
##   ttMetalTiled   MSL, register tiles: each thread owns a (BM/8) x 4 block (any Metal GPU)
##   ttVulkanTiled  GLSL, the same register tiles (any Vulkan device)
##
## Every target uses the same grid: ceil(rows / BM) * ceil(N / 64) threadgroups of 128 threads,
## threadgroup t covering rows (t / ceil(N/64)) * BM and columns (t mod ceil(N/64)) * 64; BM is the
## row tile (16, 32 or 64) chosen per dispatch. Slots follow the micro-VM: out = buffer 1 / binding
## 0, packed weights = 2 / 1, decode table = 3 / 2, activations = 4 / 3. A program with M = 0
## reads the row count from buffer 5 (Metal) or a push constant (Vulkan) at each dispatch.
##
## The legacy shape op OP_FP4_COOP_GEMM (M, K, N, 32, K/32: the FP4-KM blob with its decode
## table after the scale bytes in one buffer) expands to the same program (fp4TileProgram).
##
## On the cooperative-matrix target the K loop is register double-buffered: TILE_K_BEGIN fetches
## the first slice into registers, TILE_LOAD_A / the weight op write the registers to threadgroup
## memory, and TILE_MMA issues the global loads of the next slice before its multiply-adds, so the
## memory latency of slice k+1 hides behind the tensor-core work of slice k.

import std/strutils
import ./session_vm

type
  TileTarget* = enum
    ttVulkanCoop, ttVulkanTiled, ttMetalSimd, ttMetalTiled

  TileGemmShape* = object
    format*: uint32
    M*, K*, N*: int      ## M = 0: the row count is a dispatch parameter
    scaleBase*: int      ## byte of the first scale byte in the weight slot
    lutBase*: int        ## float of the 272-float decode table in the table slot

const
  TileGemmThreads* = 128      ## threads (invocations) per threadgroup on every target
  TileGemmCols* = 64          ## output columns per threadgroup
  TileGemmRowTiles* = [64, 32, 16]

proc tileTargetDepth*(t: TileTarget): int =
  ## K values one TILE_K_BEGIN step covers: two 32-value groups for the cooperative matrices
  ## (16-deep fragments, one uvec4 per group and column), one group elsewhere.
  if t == ttVulkanCoop: 64 else: 32

proc isTileGemmProgram*(p: VmProgram): bool =
  p.words.len > 0 and (p.words[0] and 0xFF'u32) == OP_TILE_GEMM

proc fp4TileProgram*(p: VmProgram): VmProgram =
  ## The tile GEMM program of an FP4-KM micro-kernel: unchanged for an OP_TILE_GEMM program; the
  ## legacy OP_FP4_COOP_GEMM shape op (one blob: nibbles, scales, then the table) expands to it.
  if p.isTileGemmProgram: return p
  if p.words.len == 7 and (p.words[0] and 0xFF'u32) == OP_FP4_COOP_GEMM and p.words[6] == OP_HALT:
    let M = int(p.words[1])
    let K = int(p.words[2])
    let N = int(p.words[3])
    if p.words[4] != 32'u32 or K mod 32 != 0 or int(p.words[5]) != K div 32:
      raise newException(ValueError, "FP4 cooperative micro-kernel needs group 32 and K/32 groups")
    return buildFp4TileGemmProgram(M, K, N, N * (K div 2), fp4CoopLutWord(K, N))
  raise newException(ValueError, "not an FP4-KM tile GEMM micro-kernel program")

proc tileGemmShape*(p: VmProgram): TileGemmShape =
  let q = fp4TileProgram(p)
  if q.words.len < 6:
    raise newException(ValueError, "truncated tile GEMM header")
  result = TileGemmShape(format: vmRegA(q.words[0]), M: int(q.words[1]), K: int(q.words[2]),
    N: int(q.words[3]), scaleBase: int(q.words[4]), lutBase: int(q.words[5]))
  if result.format notin [TileFormatFp4Km, TileFormatF16]:
    raise newException(ValueError, "tile GEMM weight format " & $result.format &
      " is not supported (0 = FP4-KM, 1 = FP16)")
  if result.K <= 0 or result.K mod 32 != 0 or result.N <= 0:
    raise newException(ValueError, "unsupported tile GEMM shape K=" & $result.K & " N=" & $result.N)

proc tileGemmSupported*(s: TileGemmShape; t: TileTarget): bool =
  ## Shapes a target handles: whole K slices, and for the cooperative matrices at least one
  ## 16-column fragment. FP16 weights have Vulkan lowerings only.
  s.K mod tileTargetDepth(t) == 0 and (t != ttVulkanCoop or s.N >= 16) and
    (s.format == TileFormatFp4Km or t in [ttVulkanCoop, ttVulkanTiled])

proc tileGemmRowTile*(rows: int): int =
  ## Row tile of a session dispatch: the shortest tile that holds `rows`, at most 64.
  if rows <= 16: 16 elif rows <= 32: 32 else: 64

proc tileGemmGroups*(rows, N, rowTile: int): int =
  ## Threadgroups of one dispatch over `rows` rows.
  ((rows + rowTile - 1) div rowTile) * ((N + TileGemmCols - 1) div TileGemmCols)

# ── Lowering ─────────────────────────────────────────────────────────

type TileEmitter = object
  t: TileTarget
  s: TileGemmShape
  bm, frags: int
  lines: seq[string]
  zeroed, inLoop, loaded, decoded, stored: bool

proc glsl(e: TileEmitter): bool = e.t in [ttVulkanCoop, ttVulkanTiled]

proc add(e: var TileEmitter; text: string) =
  for line in text.strip(leading = false).splitLines:
    e.lines.add line

proc constant(e: TileEmitter; name: string; value: int): string =
  if e.glsl: "const uint " & name & " = " & $value & "u;"
  else: "constant constexpr uint " & name & " = " & $value & "u;"

proc emitHeader(e: var TileEmitter; kernelName: string) =
  let s = e.s
  let depth = tileTargetDepth(e.t)
  let nTiles = (s.N + TileGemmCols - 1) div TileGemmCols
  var consts = @[e.constant("T_K", s.K), e.constant("T_N", s.N), e.constant("T_NG", s.K div 32),
    e.constant("ROW_BYTES", s.K div 2), e.constant("SCALE_BASE", s.scaleBase),
    e.constant("LUT_BASE", s.lutBase), e.constant("BM", e.bm), e.constant("BN", TileGemmCols),
    e.constant("BK", depth), e.constant("THREADS", TileGemmThreads), e.constant("N_TILES", nTiles)]
  if s.M > 0: consts.add e.constant("T_M", s.M)
  case e.t
  of ttVulkanCoop:
    consts.add e.constant("FRAGS", e.frags)
    consts.add e.constant("LDS", depth + 8)          # padded FP16 row: 144 bytes, 16-byte aligned
    consts.add e.constant("ROW_TILES", e.bm div 16)
    consts.add e.constant("A_REGS", e.bm * (depth div 4) div TileGemmThreads)  # vec4 per invocation
  of ttMetalSimd:
    consts.add e.constant("ALD", depth + 4)
    consts.add e.constant("BLD", TileGemmCols + 4)
    consts.add e.constant("ROW_FRAGS", e.bm div 8)
  of ttMetalTiled, ttVulkanTiled:
    consts.add e.constant("TM", e.bm div 8)           # 8 thread rows x 16 thread columns
    consts.add e.constant("TN", 4)
    consts.add e.constant("NTX", 16)
    consts.add e.constant("ASTR", e.bm + 1)
    consts.add e.constant("BSTR", TileGemmCols + 1)
  let formatName = if s.format == TileFormatF16: "FP16" else: "FP4-KM"
  let note = "// Generated by the micro-VM JIT from an " & formatName & " tile GEMM program (" & $e.t &
    ", row tile " & $e.bm & "). Do not edit: change the program or tile_gemm.nim."
  # FP4-KM stages its decode table in threadgroup memory: the 16 values, plus on the cooperative
  # target the 256 scales too (the prefetched scale byte is looked up there).
  let lutLen = if s.format == TileFormatF16: 0 elif e.t == ttVulkanCoop: 272 else: 16
  let rowsDecl = if s.M > 0: "const uint ROWS = T_M;" else: "const uint ROWS = " &
    (if e.glsl: "tileRows.rows;" else: "rowsArg;")
  if e.glsl:
    var head = "#version 450\n"
    if e.t == ttVulkanCoop:
      head.add "#extension GL_KHR_cooperative_matrix : require\n" &
        "#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require\n" &
        "#extension GL_KHR_shader_subgroup_basic : require\n" &
        "#extension GL_KHR_memory_scope_semantics : require\n" &
        "#extension GL_EXT_control_flow_attributes : require\n"
    head.add note & "\n// kernel " & kernelName & "\n"
    head.add (if e.t == ttVulkanCoop: "layout(local_size_x_id = 0) in;\n" else: "layout(local_size_x = 128) in;\n")
    head.add "layout(std430, set=0, binding=0) buffer MemF32Buf { float memF32[]; };\n"
    head.add "layout(std430, set=0, binding=1) readonly buffer MemU8Words { uint memW[]; };\n"
    if e.t == ttVulkanCoop:
      head.add "layout(std430, set=0, binding=1) readonly buffer MemU8Vec { uvec4 memV[]; };\n"
    head.add "layout(std430, set=0, binding=2) readonly buffer LutBuf { float lut[]; };\n"
    head.add (if e.t == ttVulkanCoop: "layout(std430, set=0, binding=3) readonly buffer InputBuf { vec4 in4[]; };\n"
              else: "layout(std430, set=0, binding=3) readonly buffer InputBuf { float inF32[]; };\n")
    if s.M == 0:
      head.add "layout(push_constant) uniform TileRows { uint rows; } tileRows;\n"
    e.add head & consts.join("\n")
    if e.t == ttVulkanCoop:
      e.add """
shared float16_t aTile[BM * LDS];
shared float16_t bTile[BN * LDS];
shared float cStage[4u * 256u];"""
    else:
      e.add """
shared float As[BK * ASTR];
shared float Bs[BK * BSTR];"""
    if lutLen > 0: e.add "shared float lutS[" & $lutLen & "];"
    e.add """
uint byteAt(uint i) { return (memW[i >> 2] >> ((i & 3u) * 8u)) & 255u; }
void main() {
    uint lid = gl_LocalInvocationID.x;
    uint tgid = gl_WorkGroupID.x;"""
    if e.t == ttVulkanCoop:
      e.add """
    uint sg = gl_SubgroupID;
    uint lane = gl_SubgroupInvocationID;
    uint subgroupCount = gl_NumSubgroups;
    uint threads = gl_WorkGroupSize.x;"""
  else:
    var head = "#include <metal_stdlib>\n"
    if e.t == ttMetalSimd: head.add "#include <metal_simdgroup_matrix>\n"
    head.add "using namespace metal;\n" & note & "\n"
    e.add head & consts.join("\n")
    var params = @["device float*       memF32 [[buffer(1)]]",
      "device const uchar* memU8  [[buffer(2)]]",
      "device const float* lut    [[buffer(3)]]",
      "device const float* inF32  [[buffer(4)]]"]
    if s.M == 0: params.add "constant uint&      rowsArg [[buffer(5)]]"
    params.add "uint tgid [[threadgroup_position_in_grid]]"
    params.add "uint lid  [[thread_index_in_threadgroup]]"
    if e.t == ttMetalSimd:
      params.add "uint sg   [[simdgroup_index_in_threadgroup]]"
      params.add "uint lane [[thread_index_in_simdgroup]]"
    e.add "kernel void " & kernelName & "(\n    " & params.join(",\n    ") & ")\n{"
    if e.t == ttMetalSimd:
      e.add """
    threadgroup float aTile[BM * ALD];
    threadgroup float bTile[BK * BLD];
    threadgroup float cStage[4u * 64u];"""
    else:
      e.add """
    threadgroup float As[BK * ASTR];
    threadgroup float Bs[BK * BSTR];"""
    if lutLen > 0: e.add "    threadgroup float lutS[" & $lutLen & "];"
  e.add "    " & rowsDecl
  e.add """
    const uint row0 = (tgid / N_TILES) * BM;
    const uint col0 = (tgid % N_TILES) * BN;
    // Uniform per threadgroup: a grid sized for a shorter row tile has nothing past the last row.
    if (row0 >= ROWS) return;"""
  case e.t
  of ttVulkanCoop:
    e.add """
    const uint liveTiles = min(ROW_TILES, (ROWS - row0 + 15u) / 16u);
    const uint bCol = lid >> 1;
    const uint bHalf = lid & 1u;
    const uint gCol = col0 + bCol;"""
  of ttMetalSimd:
    e.add """
    const uint liveFrags = min(ROW_FRAGS, (ROWS - row0 + 7u) / 8u);
    const uint liveRows = liveFrags * 8u;
    const uint bCol = lid >> 1;
    const uint bHalf = lid & 1u;
    const uint gCol = col0 + bCol;
    const uint sgCol = sg * 16u;"""
  of ttMetalTiled, ttVulkanTiled:
    e.add """
    const uint tx = lid % NTX;
    const uint ty = lid / NTX;"""
  if lutLen == 16:
    e.add "    if (lid < 16u) lutS[lid] = lut[LUT_BASE + lid];"
  elif lutLen > 0:
    e.add "    for (uint i = lid; i < " & $lutLen & "u; i += THREADS) lutS[i] = lut[LUT_BASE + i];"

proc emitBarrier(e: var TileEmitter) =
  e.add (if e.glsl: "        barrier();" else: "        threadgroup_barrier(mem_flags::mem_threadgroup);")

proc emitZero(e: var TileEmitter) =
  if e.zeroed or e.inLoop: raise newException(ValueError, "tile GEMM: TILE_ZERO must come once, before the K loop")
  e.zeroed = true
  case e.t
  of ttVulkanCoop:
    e.add """
    coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> acc[FRAGS * ROW_TILES];
    [[unroll]] for (uint i = 0u; i < FRAGS * ROW_TILES; ++i)
        acc[i] = coopmat<float, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(0.0);
    barrier();"""
  of ttMetalSimd:
    e.add """
    simdgroup_float8x8 acc[2u * ROW_FRAGS];
    for (uint i = 0u; i < 2u * ROW_FRAGS; ++i)
        acc[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    threadgroup_barrier(mem_flags::mem_threadgroup);"""
  of ttMetalTiled:
    e.add """
    float acc[TM][TN];
    for (uint i = 0u; i < TM; ++i)
        for (uint j = 0u; j < TN; ++j) acc[i][j] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);"""
  of ttVulkanTiled:
    e.add """
    float acc[TM * TN];
    for (uint i = 0u; i < TM * TN; ++i) acc[i] = 0.0;
    barrier();"""

proc coopFetch(e: TileEmitter; nb, indent: string): string =
  ## Cooperative target: global loads of the K slice starting at `nb` into the staging registers
  ## (activations: A_REGS vec4 per invocation; weights: one 32-value group per invocation).
  result = indent & "[[unroll]] for (uint j = 0u; j < A_REGS; ++j) {\n" &
    indent & "    uint i = lid + j * THREADS;\n" &
    indent & "    uint row = row0 + (i >> 4);\n" &
    indent & "    aReg[j] = row < ROWS ? in4[row * (T_K / 4u) + (" & nb & " >> 2) + (i & 15u)] : vec4(0.0);\n" &
    indent & "}\n"
  if e.s.format == TileFormatF16:
    result.add indent & "[[unroll]] for (uint j = 0u; j < 4u; ++j)\n" &
      indent & "    bH[j] = gCol < T_N ? memV[gCol * (T_K / 8u) + (" & nb & " >> 3) + bHalf * 4u + j] : uvec4(0u);\n"
  else:
    result.add indent & "if (gCol < T_N) {\n" &
      indent & "    bReg = memV[gCol * T_NG + (" & nb & " >> 5) + bHalf];\n" &
      indent & "    bSc = byteAt(SCALE_BASE + gCol * T_NG + (" & nb & " >> 5) + bHalf);\n" &
      indent & "}\n"

proc emitKBegin(e: var TileEmitter) =
  if not e.zeroed or e.inLoop: raise newException(ValueError, "tile GEMM: TILE_K_BEGIN needs TILE_ZERO first and does not nest")
  e.inLoop = true
  if e.t == ttVulkanCoop:
    e.add "    // Staging registers, loaded one slice ahead of the multiply-adds."
    e.add "    vec4 aReg[A_REGS];"
    if e.s.format == TileFormatF16:
      e.add "    uvec4 bH[4];"
    else:
      e.add "    uvec4 bReg = uvec4(0u);\n    uint bSc = 0u;"
    e.add e.coopFetch("0u", "    ")
  e.add "    for (uint base = 0u; base < T_K; base += BK) {"
  e.add (if e.glsl: "        uint g = base >> 5;" else: "        const uint g = base >> 5;")

proc emitKEnd(e: var TileEmitter) =
  if not e.inLoop: raise newException(ValueError, "tile GEMM: TILE_K_END without TILE_K_BEGIN")
  if not (e.loaded and e.decoded): raise newException(ValueError, "tile GEMM: the K loop must stage activations and weights")
  e.inLoop = false
  e.add "    }"

proc emitLoadA(e: var TileEmitter) =
  if not e.inLoop: raise newException(ValueError, "tile GEMM: TILE_LOAD_A outside the K loop")
  e.loaded = true
  case e.t
  of ttVulkanCoop:
    e.add """
        // Activations: the prefetched slice, 16 vec4 per row (rows past the end are zero).
        [[unroll]] for (uint j = 0u; j < A_REGS; ++j) {
            uint i = lid + j * THREADS;
            uint o = (i >> 4) * LDS + (i & 15u) * 4u;
            vec4 v = aReg[j];
            aTile[o] = float16_t(v.x);
            aTile[o + 1u] = float16_t(v.y);
            aTile[o + 2u] = float16_t(v.z);
            aTile[o + 3u] = float16_t(v.w);
        }"""
  of ttMetalSimd:
    e.add """
        // Activations: the live 8-row fragments, 8 float4 per row (rows past the end are zero).
        for (uint i = lid; i < liveRows * (BK / 4u); i += THREADS) {
            const uint r = i >> 3;
            const uint q = i & 7u;
            const uint row = row0 + r;
            const float4 v = row < ROWS
                ? float4(*((device const packed_float4*)(inF32 + row * T_K + base + q * 4u)))
                : float4(0.0f);
            const uint o = r * ALD + q * 4u;
            aTile[o] = v.x;
            aTile[o + 1u] = v.y;
            aTile[o + 2u] = v.z;
            aTile[o + 3u] = v.w;
        }"""
  of ttMetalTiled:
    e.add """
        // Activations, transposed into [k][row] so each thread reads its TM rows side by side.
        for (uint i = lid; i < BM * BK; i += THREADS) {
            const uint mm = i / BK;
            const uint kk = i % BK;
            const uint row = row0 + mm;
            As[kk * ASTR + mm] = row < ROWS ? inF32[row * T_K + base + kk] : 0.0f;
        }"""
  of ttVulkanTiled:
    e.add """
        // Activations, transposed into [k][row] so each thread reads its TM rows side by side.
        for (uint i = lid; i < BM * BK; i += THREADS) {
            uint mm = i / BK;
            uint kk = i % BK;
            uint row = row0 + mm;
            As[kk * ASTR + mm] = row < ROWS ? inF32[row * T_K + base + kk] : 0.0;
        }"""

proc emitLoadBF16(e: var TileEmitter) =
  if not e.inLoop: raise newException(ValueError, "tile GEMM: TILE_LOAD_B_F16 outside the K loop")
  if e.s.format != TileFormatF16:
    raise newException(ValueError, "tile GEMM: TILE_LOAD_B_F16 needs weight format 1 (FP16)")
  e.decoded = true
  case e.t
  of ttVulkanCoop:
    e.add """
        // Weights: invocation (column, half) stages 32 prefetched halves (four uvec4) of its column.
        {
            uint ob = bCol * LDS + bHalf * 32u;
            [[unroll]] for (uint j = 0u; j < 4u; ++j) {
                [[unroll]] for (uint q = 0u; q < 4u; ++q) {
                    f16vec2 h = unpackFloat2x16(bH[j][q]);
                    bTile[ob + 8u * j + 2u * q] = h.x;
                    bTile[ob + 8u * j + 2u * q + 1u] = h.y;
                }
            }
        }"""
  of ttVulkanTiled:
    e.add """
        // Weights: one 32-bit word holds two consecutive halves of a column.
        for (uint i = lid; i < BN * 16u; i += THREADS) {
            uint nn = i >> 4;
            uint j = i & 15u;
            uint col = col0 + nn;
            vec2 v = vec2(0.0);
            if (col < T_N) v = unpackHalf2x16(memW[(col * T_K + base) / 2u + j]);
            Bs[(2u * j) * BSTR + nn] = v.x;
            Bs[(2u * j + 1u) * BSTR + nn] = v.y;
        }"""
  of ttMetalSimd, ttMetalTiled:
    raise newException(ValueError, "tile GEMM: FP16 weights have no Metal lowering")

proc emitDecodeFp4(e: var TileEmitter) =
  if not e.inLoop: raise newException(ValueError, "tile GEMM: TILE_DECODE_FP4 outside the K loop")
  if e.s.format != TileFormatFp4Km:
    raise newException(ValueError, "tile GEMM: TILE_DECODE_FP4 needs weight format 0 (FP4-KM)")
  e.decoded = true
  case e.t
  of ttVulkanCoop:
    e.add """
        // Weights: invocation (column, half) decodes its prefetched 32-value group (one uvec4 +
        // one scale byte, looked up in the staged table).
        {
            uvec4 packed = bReg;
            float scale = gCol < T_N ? lutS[16u + bSc] : 0.0;
            uint ob = bCol * LDS + bHalf * 32u;
            [[unroll]] for (uint j = 0u; j < 4u; ++j) {
                uint word = packed[j];
                [[unroll]] for (uint b = 0u; b < 4u; ++b) {
                    uint byteV = (word >> (8u * b)) & 255u;
                    bTile[ob + 4u * j + b] = float16_t(lutS[byteV & 15u] * scale);
                    bTile[ob + 16u + 4u * j + b] = float16_t(lutS[byteV >> 4] * scale);
                }
            }
        }"""
  of ttMetalSimd:
    e.add """
        // Weights: thread (column, half) decodes 16 values of the group: half 0 the low nibbles
        // (elements 0..15), half 1 the high nibbles (16..31).
        {
            uint words[4] = {0u, 0u, 0u, 0u};
            float scale = 0.0f;
            if (gCol < T_N) {
                device const uint* src = (device const uint*)(memU8 + gCol * ROW_BYTES + g * 16u);
                words[0] = src[0];
                words[1] = src[1];
                words[2] = src[2];
                words[3] = src[3];
                scale = lut[LUT_BASE + 16u + uint(memU8[SCALE_BASE + gCol * T_NG + g])];
            }
            const uint shift = bHalf * 4u;
            const uint kOff = bHalf * 16u;
            for (uint j = 0u; j < 4u; ++j) {
                const uint word = words[j];
                for (uint b = 0u; b < 4u; ++b) {
                    const uint nibble = (word >> (8u * b + shift)) & 15u;
                    bTile[(kOff + 4u * j + b) * BLD + bCol] = lutS[nibble] * scale;
                }
            }
        }"""
  of ttMetalTiled:
    e.add """
        // Weights: one packed byte holds elements j (low nibble) and j+16 (high nibble) of the
        // group, so a thread decodes both from one byte read and one scale lookup.
        for (uint i = lid; i < BN * 16u; i += THREADS) {
            const uint nn = i >> 4;
            const uint j = i & 15u;
            const uint col = col0 + nn;
            float lo = 0.0f, hi = 0.0f;
            if (col < T_N) {
                const uint byteV = memU8[col * ROW_BYTES + g * 16u + j];
                const float sc = lut[LUT_BASE + 16u + uint(memU8[SCALE_BASE + col * T_NG + g])];
                lo = lutS[byteV & 15u] * sc;
                hi = lutS[byteV >> 4] * sc;
            }
            Bs[j * BSTR + nn] = lo;
            Bs[(j + 16u) * BSTR + nn] = hi;
        }"""
  of ttVulkanTiled:
    e.add """
        // Weights: one packed byte holds elements j (low nibble) and j+16 (high nibble) of the
        // group, so an invocation decodes both from one byte read and one scale lookup.
        for (uint i = lid; i < BN * 16u; i += THREADS) {
            uint nn = i >> 4;
            uint j = i & 15u;
            uint col = col0 + nn;
            float lo = 0.0, hi = 0.0;
            if (col < T_N) {
                uint byteV = byteAt(col * ROW_BYTES + g * 16u + j);
                float sc = lut[LUT_BASE + 16u + byteAt(SCALE_BASE + col * T_NG + g)];
                lo = lutS[byteV & 15u] * sc;
                hi = lutS[byteV >> 4] * sc;
            }
            Bs[j * BSTR + nn] = lo;
            Bs[(j + 16u) * BSTR + nn] = hi;
        }"""

proc emitMma(e: var TileEmitter) =
  if not (e.inLoop and e.loaded and e.decoded):
    raise newException(ValueError, "tile GEMM: TILE_MMA needs both staged slices inside the K loop")
  case e.t
  of ttVulkanCoop:
    e.add "        if (base + BK < T_K) {\n            uint nb = base + BK;"
    e.add e.coopFetch("nb", "            ")
    e.add "        }"
    e.add """
        [[unroll]] for (uint kk = 0u; kk < BK; kk += 16u) {
            [[unroll]] for (uint f = 0u; f < FRAGS; ++f) {
                uint colFragment = sg + f * subgroupCount;
                if (colFragment < 4u) {
                    coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB> bm;
                    coopMatLoad(bm, bTile, colFragment * 16u * LDS + kk, LDS,
                        gl_CooperativeMatrixLayoutColumnMajor);
                    [[unroll]] for (uint r = 0u; r < ROW_TILES; ++r) {
                        if (r < liveTiles) {
                            coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA> am;
                            coopMatLoad(am, aTile, r * 16u * LDS + kk, LDS,
                                gl_CooperativeMatrixLayoutRowMajor);
                            acc[f * ROW_TILES + r] = coopMatMulAdd(am, bm, acc[f * ROW_TILES + r]);
                        }
                    }
                }
            }
        }"""
  of ttMetalSimd:
    e.add """
        for (uint kk = 0u; kk < BK; kk += 8u) {
            simdgroup_float8x8 b0, b1;
            simdgroup_load(b0, bTile + kk * BLD + sgCol, BLD);
            simdgroup_load(b1, bTile + kk * BLD + sgCol + 8u, BLD);
            for (uint r = 0u; r < ROW_FRAGS; ++r) {
                if (r < liveFrags) {
                    simdgroup_float8x8 am;
                    simdgroup_load(am, aTile + r * 8u * ALD + kk, ALD);
                    simdgroup_multiply_accumulate(acc[2u * r], am, b0, acc[2u * r]);
                    simdgroup_multiply_accumulate(acc[2u * r + 1u], am, b1, acc[2u * r + 1u]);
                }
            }
        }"""
  of ttMetalTiled:
    e.add """
        for (uint kk = 0u; kk < BK; ++kk) {
            float ra[TM], rb[TN];
            for (uint i = 0u; i < TM; ++i) ra[i] = As[kk * ASTR + ty * TM + i];
            for (uint j = 0u; j < TN; ++j) rb[j] = Bs[kk * BSTR + tx * TN + j];
            for (uint i = 0u; i < TM; ++i)
                for (uint j = 0u; j < TN; ++j) acc[i][j] = fma(ra[i], rb[j], acc[i][j]);
        }"""
  of ttVulkanTiled:
    e.add """
        for (uint kk = 0u; kk < BK; ++kk) {
            float ra[TM];
            float rb[TN];
            for (uint i = 0u; i < TM; ++i) ra[i] = As[kk * ASTR + ty * TM + i];
            for (uint j = 0u; j < TN; ++j) rb[j] = Bs[kk * BSTR + tx * TN + j];
            for (uint i = 0u; i < TM; ++i)
                for (uint j = 0u; j < TN; ++j) acc[i * TN + j] = fma(ra[i], rb[j], acc[i * TN + j]);
        }"""

proc emitStore(e: var TileEmitter) =
  if e.inLoop or not e.zeroed or e.stored:
    raise newException(ValueError, "tile GEMM: TILE_STORE must come once, after the K loop")
  e.stored = true
  case e.t
  of ttVulkanCoop:
    e.add """
    // Each subgroup stages its own 16x16 tile, then writes the in-bounds elements.
    [[unroll]] for (uint f = 0u; f < FRAGS; ++f) {
        uint colFragment = sg + f * subgroupCount;
        if (colFragment < 4u) {
            [[unroll]] for (uint r = 0u; r < ROW_TILES; ++r) {
                if (r < liveTiles) {
                    coopMatStore(acc[f * ROW_TILES + r], cStage, colFragment * 256u, 16,
                        gl_CooperativeMatrixLayoutRowMajor);
                    subgroupMemoryBarrierShared();
                    subgroupBarrier();
                    for (uint e = lane; e < 256u; e += gl_SubgroupSize) {
                        uint row = row0 + r * 16u + (e >> 4);
                        uint col = col0 + colFragment * 16u + (e & 15u);
                        if (row < ROWS && col < T_N)
                            memF32[row * T_N + col] = cStage[colFragment * 256u + e];
                    }
                    subgroupBarrier();
                }
            }
        }
    }"""
  of ttMetalSimd:
    e.add """
    // Full 8x8 fragments go straight to memory; edge fragments are staged per simdgroup.
    for (uint r = 0u; r < ROW_FRAGS; ++r) {
        if (r >= liveFrags) continue;
        for (uint f = 0u; f < 2u; ++f) {
            const uint row = row0 + r * 8u;
            const uint col = col0 + sgCol + f * 8u;
            if (col >= T_N) continue;
            if (row + 8u <= ROWS && col + 8u <= T_N) {
                simdgroup_store(acc[2u * r + f], memF32 + row * T_N + col, ulong(T_N));
            } else {
                threadgroup float* stage = cStage + sg * 64u;
                simdgroup_store(acc[2u * r + f], stage, 8);
                simdgroup_barrier(mem_flags::mem_threadgroup);
                for (uint e = lane; e < 64u; e += 32u) {
                    const uint rr = row + (e >> 3);
                    const uint cc = col + (e & 7u);
                    if (rr < ROWS && cc < T_N) memF32[rr * T_N + cc] = stage[e];
                }
                simdgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
    }"""
  of ttMetalTiled:
    e.add """
    for (uint i = 0u; i < TM; ++i) {
        const uint row = row0 + ty * TM + i;
        if (row >= ROWS) continue;
        for (uint j = 0u; j < TN; ++j) {
            const uint col = col0 + tx * TN + j;
            if (col < T_N) memF32[row * T_N + col] = acc[i][j];
        }
    }"""
  of ttVulkanTiled:
    e.add """
    for (uint i = 0u; i < TM; ++i) {
        uint row = row0 + ty * TM + i;
        if (row >= ROWS) continue;
        for (uint j = 0u; j < TN; ++j) {
            uint col = col0 + tx * TN + j;
            if (col < T_N) memF32[row * T_N + col] = acc[i * TN + j];
        }
    }"""

proc lowerTileGemm*(p: VmProgram; target: TileTarget; kernelName: string; rowTile: int;
                    fragments = 2): string =
  ## Source of the tile GEMM micro-kernel `p` (or a legacy FP4 cooperative shape op) for one
  ## target and row tile. `fragments` (cooperative matrices only): 16-column fragments per
  ## subgroup; 1 needs four 32-wide subgroups per 128-invocation workgroup (NVIDIA), 2 also covers
  ## 64-wide subgroups.
  if rowTile notin TileGemmRowTiles:
    raise newException(ValueError, "tile GEMM row tile must be 16, 32 or 64")
  if fragments notin [1, 2]:
    raise newException(ValueError, "tile GEMM fragments per subgroup must be 1 or 2")
  let q = fp4TileProgram(p)
  let s = tileGemmShape(q)
  if not tileGemmSupported(s, target):
    raise newException(ValueError, "tile GEMM K=" & $s.K & " N=" & $s.N & " does not fit " & $target)
  var e = TileEmitter(t: target, s: s, bm: rowTile, frags: fragments)
  e.emitHeader(kernelName)
  var pc = 1 + opImmediateWords(OP_TILE_GEMM)
  var halted = false
  while pc < q.words.len:
    let op = q.words[pc] and 0xFF'u32
    if halted: raise newException(ValueError, "tile GEMM: ops after HALT")
    case op
    of OP_TILE_ZERO: e.emitZero()
    of OP_TILE_K_BEGIN: e.emitKBegin()
    of OP_TILE_K_END: e.emitKEnd()
    of OP_TILE_LOAD_A: e.emitLoadA()
    of OP_TILE_DECODE_FP4: e.emitDecodeFp4()
    of OP_TILE_LOAD_B_F16: e.emitLoadBF16()
    of OP_TILE_MMA: e.emitMma()
    of OP_BARRIER_TG:
      if not e.inLoop: raise newException(ValueError, "tile GEMM: barriers belong inside the K loop")
      e.emitBarrier()
    of OP_TILE_STORE: e.emitStore()
    of OP_NOP: discard
    of OP_HALT: halted = true
    else:
      raise newException(ValueError, "tile GEMM: op 0x" & toHex(op, 2) & " has no tile lowering")
    pc += 1 + opImmediateWords(op)
  if not (halted and e.stored):
    raise newException(ValueError, "tile GEMM program must store its tile and end with HALT")
  e.lines.add "}"
  if e.glsl:
    # GLSL `const` needs a constant initializer: locals of main are plain variables.
    for line in e.lines.mitems:
      if line.startsWith("    "): line = line.replace("const uint ", "uint ")
  e.lines.join("\n") & "\n"

proc tileGemmReference*(x: openArray[float32]; packed: openArray[uint8]; table: openArray[float32];
                        rows, K, N, scaleBase, lutBase: int): seq[float32] =
  ## Host reference of the FP4-KM tile GEMM (the CPU VM computes the same sums in C).
  result = newSeq[float32](rows * N)
  let groups = K div 32
  for r in 0 ..< rows:
    for c in 0 ..< N:
      var sum = 0'f32
      for g in 0 ..< groups:
        let scale = table[lutBase + 16 + int(packed[scaleBase + c * groups + g])]
        for j in 0 ..< 16:
          let b = packed[c * (K div 2) + g * 16 + j]
          sum += x[r * K + g * 32 + j] * (table[lutBase + int(b and 15)] * scale)
          sum += x[r * K + g * 32 + 16 + j] * (table[lutBase + int(b shr 4)] * scale)
      result[r * N + c] = sum
