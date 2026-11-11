# bybylang.nim - BybyLang AOT executable + Nim code generation + auto compile release
# Functions: "function NAME" or "function NAME(a: int, b = 2): T {.pragma.}"
# opens a definition; its body is the block indented under that line.
#
# --------------------------
# "session:" syntax - uses the Session API of gpubackend.nim
# --------------------------
# gpubackend.nim provides a Session API (sessionBegin/sessionUpload/
# sessionMatmul/sessionVecOp/sessionActivation/sessionSoftmax/
# sessionLayernormEnc/sessionRope/sessionEnd/sessionRead/sessionFree) that batches
# several GPU commands into one session (fewer commit+wait round trips than separate
# "gpu ..." commands). Syntax:
#
#   session:
#     gpu add A, B -> C size 4
#     gpu mul C, D -> E size 4
#     gpu relu E -> F
#     gpu fused_add_act A, B -> G act "relu"
#     gpu rope Q, COS, SIN -> Q2 seqlen 4 heads 1 dim 4
#
# Ops ENCODED INSIDE the session: add/sub/mul/div, relu/sigmoid/tanh,
# fused_add_act (sessionVecOp("add") + sessionActivation(act) encoded DIRECTLY
# into the SAME enclosing session, no nested session), softmax, matmul,
# layernorm, rope. If sessionBegin() fails (the backend has no session
# support), the whole block falls back to plain "gpu ..." calls
# (immediate - still the real gpuXxx() GPU functions, NEVER the CPU),
# so the result is always correct. The other GPU ops
# (embedding, attention, attention_backward, layernorm_backward, apflu...)
# have NO session variant in gpubackend.nim, so they are still generated
# as immediate calls; they need not be inside "session:" (inside it
# they still work, they are just not batched).
import strutils, os, osproc, tables, sequtils, times
import ./gpubackend
import ./backends/cpu/cpu_backend as cpu
import ./micro_kernel
import ./backends/metal/metal_backend
import ./backends/vulkan/vulkan_driver
import ./diag as dg
export gpubackend, micro_kernel, metal_backend, vulkan_driver

# --------------------------
# Traceback system - LUÔN BẬT, IN RA TERMINAL
# --------------------------
type
  TraceLevel* = enum
    tlDebug, tlInfo, tlWarn, tlError

var gTraceEnabled = true
var gTraceLevel = tlWarn   # -v: info, -vv: debug

proc currentTime(): string =
  let t = getTime()
  return format(t, "HH:mm:ss")

proc trace*(level: TraceLevel, msg: string, file: string = "", line: int = 0) =
  if not gTraceEnabled: return
  if level < gTraceLevel: return
  
  let prefix = case level
    of tlDebug: "[DEBUG]"
    of tlInfo: "[INFO]"
    of tlWarn: "[WARN]"
    of tlError: "[ERROR]"
  
  let loc = if file.len > 0: file else: "bybylang.nim"
  let ln = if line > 0: line else: instantiationInfo().line
  
  echo prefix & " " & currentTime() & " " & loc & ":" & $ln & " - " & msg

proc traceDebug*(msg: string) = trace(tlDebug, msg)
proc traceInfo*(msg: string) = trace(tlInfo, msg)
proc traceWarn*(msg: string) = trace(tlWarn, msg)
proc traceError*(msg: string) = trace(tlError, msg)

# --------------------------
# Helpers
# --------------------------
proc stripQuotes(s: string): string =
  if s.len >= 2 and s[0] == '"' and s[^1] == '"':
    return s[1..^2]
  else:
    return s
    
proc parseIntSafe(s: string): int =
  try:
    return parseInt(s)
  except:
    return 0

# NEW: reads a parameter by NAME (independent of a fixed position
# like the old branches of genGpuLine) - used by the newly added ops.
# e.g. rhsParts = ["Y", "n", "10", "p", "0.5", "seed", "42"],
# findNamedParam(rhsParts, "p", "0.0") -> "0.5"
proc findNamedParam*(parts: seq[string], key: string, default: string): string =
  for i in 0..<parts.len:
    if parts[i] == key and i+1 < parts.len:
      return parts[i+1]
  return default

# --------------------------
# Types
# --------------------------
type
  Mode = enum
    Low, Mid, High

  Token = object
    sym: string
    text: string
    indent: int
    file: string   ## source file the line came from (as given on the command line / import)
    line: int      ## 1-based line number in `file`
    
# --------------------------
# Simulated RAM / bus / pins
# --------------------------
const RAM_SIZE = 1024
var RAM: array[0..RAM_SIZE-1, int]
var BUS: seq[string] = @[]
var Pins: array[0..31, bool]

var ignoreErrors = false
var quietMode = false

# function table holding body tokens
var funcTable = initTable[string, seq[Token]]()
proc tokenizeLine(line: string): Token =
  var tok: Token
  tok.indent = line.len - line.strip(chars={' ', '\t'}).len
  let clean = line.strip()

  if clean.len == 0:
    tok.sym = "empty"
    tok.text = ""
  elif clean.startsWith("#"):
    tok.sym = "comment"
    tok.text = clean
  elif clean.startsWith("micro kernel "):
    tok.sym = "microkernel"
    tok.text = clean["micro kernel ".len..^1]
  elif clean.startsWith("function "):
    tok.sym = "function"
    tok.text = clean["function ".len..^1].strip()
  elif clean.startsWith("import nim "):
    # NEW: "import nim \"strutils\"" -> imports a REAL Nim module (not
    # a .bybylang file) - MUST be checked BEFORE the plain "import " branch,
    # otherwise that branch would take it for a .bybylang file name.
    tok.sym = "nimimport"
    tok.text = stripQuotes(clean["import nim ".len..^1].strip())
  elif clean.startsWith("import "):
    tok.sym = "import"
    tok.text = clean["import ".len..^1].strip()
  elif clean.startsWith("print "):
    tok.sym = "print"
    tok.text = clean
  else:
    tok.sym = "other"
    tok.text = clean

  return tok
# --------------------------
# Import system: "import filename" or "import \"path/name.bybylang\""
# Loaded recursively, import cycles prevented with `visited` (absolute paths).
# The import token is replaced by all tokens of the imported file (placed at
# the import position); it does not generate any Nim line itself.
# --------------------------
proc resolveImports(filename: string, visited: var seq[string]): seq[Token] =
  result = @[]
  let absPath = absolutePath(filename)
  if absPath in visited:
    traceDebug("Skipping already imported: " & filename)
    return
  visited.add(absPath)

  if not fileExists(filename):
    dg.diagEmit(dg.dsError, "", 0, 0, "file not found: " & filename)
    return

  traceInfo("Resolving imports for: " & filename)
  var lineNo = 0
  for line in lines(filename):
    inc lineNo
    var t = tokenizeLine(line)
    t.file = filename
    t.line = lineNo
    if t.sym == "import":
      var target = stripQuotes(t.text)
      if not target.endsWith(".bybylang"):
        target &= ".bybylang"
      if not fileExists(target):
        target = filename.parentDir() / target
      if not fileExists(target):
        dg.diagEmit(dg.dsError, filename, lineNo, t.indent + 1,
          "cannot find imported file '" & stripQuotes(t.text) & "'", line, line.strip().len)
        continue
      result.add(resolveImports(target, visited))
    else:
      result.add(t)
  traceDebug("Resolved " & $result.len & " tokens from " & filename)

# Reads the root .bybylang file + every imported file into one token list
proc tokenizeFile(filename: string): seq[Token] =
  traceInfo("Tokenizing file: " & filename)
  var visited: seq[string] = @[]
  let result = resolveImports(filename, visited)
  traceInfo("Total tokens: " & $result.len)
  return result

# --------------------------
# Source mapping: every generated Nim line gets a trailing `#bb:<line>:<file>` comment, so an
# error reported by Nim can be translated back to the .bybylang line that produced it.
# --------------------------
var gSrcFiles: seq[string] = @[]

proc srcFileIndex(file: string): int =
  let at = gSrcFiles.find(file)
  if at >= 0: return at
  gSrcFiles.add file
  gSrcFiles.high

proc tagLines(lines: var seq[string]; from0: int; tok: Token) =
  if tok.line <= 0: return
  let marker = " #bb:" & $tok.line & ":" & $srcFileIndex(tok.file)
  for i in max(from0, 0) ..< lines.len:
    let l = lines[i]
    if l.len == 0 or '\n' in l or " #bb:" in l or l.strip().startsWith("#"): continue
    lines[i] = l & marker

# --------------------------
# Line classification shared by the code generator.
# --------------------------
proc isTrivia(t: Token): bool =
  ## Blank lines and comments never open or close a block.
  t.sym == "empty" or t.sym == "comment"

proc nextCode(tokens: seq[Token], idx: int): int =
  ## Index of the first token at or after idx that is not trivia.
  result = idx
  while result < tokens.len and tokens[result].isTrivia: inc result

proc isNimIdent(s: string): bool =
  if s.len == 0 or s[0] notin IdentStartChars: return false
  for c in s:
    if c notin IdentChars: return false
  true

proc functionName(header: string): string =
  ## "name(a: int): T {.p.}" -> "name"
  result = header.strip()
  for i, c in result:
    if c notin IdentChars: return result[0 ..< i]

proc functionSignature(header: string): string =
  ## "name" -> "name()"; a header with a parameter list stays as written.
  let h = header.strip()
  let name = functionName(h)
  let rest = h[name.len..^1].strip()
  if rest.startsWith("("): h
  elif rest.len > 0: name & "() " & rest
  else: name & "()"

proc functionParams(header: string): seq[string] =
  ## Parameter names of "name(a, b: int; c = 1)".
  let open = header.find('(')
  let close = header.find(')')
  if open < 0 or close < open: return
  for group in header[open + 1 ..< close].split({',', ';'}):
    var name = group
    let cut = name.find({':', '='})
    if cut >= 0: name = name[0 ..< cut]
    name = name.strip()
    if name.isNimIdent: result.add name

proc declaredNames(line: string): seq[string] =
  ## Names introduced by "var a, b: T", "let x {.global.} = y", "const K = 1".
  var rest = line.split(maxsplit = 1)[^1]
  let cut = rest.find({'=', ':'})
  if cut >= 0: rest = rest[0 ..< cut]
  for part in rest.split(','):
    var name = part
    let pragma = name.find("{.")
    if pragma >= 0: name = name[0 ..< pragma]
    name = name.strip().strip(chars = {'*'})
    if name.isNimIdent: result.add name

proc isDeclaration(line: string): bool =
  line.startsWith("var ") or line.startsWith("let ") or line.startsWith("const ")

proc assignTarget(line: string): string =
  ## "name = expr" -> "name". Field/index targets ("a.b = x", "a[i] = x"),
  ## compound operators ("x += 1") and comparisons return "" and are emitted
  ## unchanged as Nim.
  let eq = line.find('=')
  if eq <= 0 or (eq + 1 < line.len and line[eq + 1] == '='): return ""
  if line[eq - 1] in {'!', '<', '>', '=', '+', '-', '*', '/', '&', '%', '|',
                      '^', '~', '.', '@', '$', ':'}:
    return ""
  let left = line[0 ..< eq].strip()
  if left.isNimIdent: left else: ""

proc diagStatement(line: string): string =
  ## "warn", "error" or "traceback" when `line` is that statement ("" otherwise).
  ## `error = 3` or `warn.x` are ordinary Nim and stay untouched.
  for word in ["warn", "error", "traceback"]:
    if line == word:
      return (if word == "traceback": word else: "")
    if line.startsWith(word & " "):
      let rest = line[word.len..^1].strip()
      if rest.len > 0 and (rest[0] in {'"', '$', '_'} or rest[0].isAlphaAscii):
        return word
  ""

proc isTypeSection(line: string): bool =
  line == "type" or line.startsWith("type ")

proc isBlockHeader(line: string): bool =
  ## Nim statements that own an indented block, besides if/elif/else/while/for.
  if not line.endsWith(":"): return false
  let word = line.split({' ', ':'})[0]
  word in ["try", "except", "finally", "of", "when", "block", "case", "defer"]

proc executeMicroKernel*(backend: GpuBackend; kernelName: string;
    program: VmProgram; inputValues: seq[float32]; outputElems, nThreads,
    tgSize: int): seq[float32] =
  ## Compile and run a BybyLang bytecode program inside one backend session.
  let selected = case backend
    of gbAuto: detectBackend()
    of gbCuda: gbVulkan
    else: backend
  let isVulkan = selected == gbVulkan
  if selected == gbCpu:
    var outv = newSeq[float32](outputElems)
    if nThreads <= 0 or tgSize <= 0 or tgSize > nThreads:
      raise newException(ValueError, "micro kernel needs positive output, threads and group sizes")
    if not cpu.vmRun(program.words, outv, inputValues, nThreads, 0, tgSize):
      raise newException(ValueError, "CPU micro kernel failed: " & cpu.cpuLastErrorMessage())
    return outv
  if selected notin {gbMetal, gbVulkan}:
    raise newException(ValueError, "micro kernel requires a Metal, Vulkan or CPU backend")
  if outputElems <= 0 or nThreads <= 0 or tgSize <= 0 or tgSize > nThreads:
    raise newException(ValueError, "micro kernel needs positive output, threads and group sizes")
  if inputValues.len == 0:
    raise newException(ValueError, "micro kernel input cannot be empty")

  if not isVulkan:
    doAssert ensureGpuInit(selected), "micro-kernel: backend initialization failed"
  let source = if isVulkan: lowerBytecodeToGlsl(program, kernelName)
               else: lowerBytecodeToMetal(program, kernelName)
  if not isVulkan:
    doAssert tryMetalJitCompile(source, kernelName),
      "micro-kernel: Metal JIT compilation failed"

  doAssert sessionBegin(selected), "micro-kernel: sessionBegin failed"
  var input = sessionUpload(selected, inputValues)
  var output = sessionAllocScratch(selected, outputElems)
  doAssert sessionEncPipeline(isVulkan, kernelName, source),
    "micro-kernel: pipeline setup failed"

  template bindBuffer(vkSlot, metalSlot: int; h: SessionHandle) =
    doAssert sessionEncBindVkOrMetal(isVulkan,
      (if isVulkan: vkSlot else: metalSlot), h.metalHandle, h.vulkanHandle),
      "micro-kernel: buffer bind failed"
  bindBuffer(0, 1, output) # output / memF32
  bindBuffer(1, 2, output) # byte view
  bindBuffer(2, 3, output) # LUT view
  bindBuffer(3, 4, input)  # input / inF32
  if isVulkan: bindBuffer(4, 5, output) # uint view

  doAssert sessionEncDispatch(isVulkan, nThreads, tgSize),
    "micro-kernel: dispatch failed"
  doAssert sessionEnd(selected), "micro-kernel: sessionEnd failed: " &
    sessionGetLastError(selected)
  result = sessionRead(selected, output, outputElems)
  sessionFree(selected, input)
  sessionFree(selected, output)

# --------------------------
# Hardware-level functions
# --------------------------
proc apuTran(name: string, payload: string) =
  traceDebug("apuTran: " & name & " -> " & payload)
  BUS.add(payload)
  if not quietMode:
    echo "[APU-TRAN] ", name, " -> ", payload

proc apuMem(action: string, target: string, value: string) =
  traceDebug("apuMem: " & action & " " & target & " " & value)
  let ramAddr = parseIntSafe(target.replace("RAM",""))
  if ramAddr < 0 or ramAddr >= RAM_SIZE:
    traceError("Invalid RAM address: " & $ramAddr)
    if not ignoreErrors:
      echo "[ERROR] Invalid RAM address: ", ramAddr
      quit(1)
    return
  if action == "write":
    RAM[ramAddr] = parseIntSafe(value)
    if not quietMode:
      echo "[APU-MEM] RAM[", ramAddr, "] <- ", value
  elif action == "read":
    if not quietMode:
      echo "[APU-MEM] RAM[", ramAddr, "] -> ", RAM[ramAddr]

proc apuCore(mode: int, code: string) =
  traceDebug("apuCore: mode=" & $mode & " code=" & code)
  if not quietMode:
    echo "[APU-CORE] Mode: ", mode, ", running: ", code

proc apuPin(pin: int, state: string) =
  traceDebug("apuPin: pin=" & $pin & " state=" & state)
  if pin < 0 or pin > 31:
    traceError("Invalid pin: " & $pin)
    if not ignoreErrors:
      echo "[ERROR] Invalid pin: ", pin
      quit(1)
    return
  Pins[pin] = (state == "high")
  if not quietMode:
    echo "[APU-PIN] pin ", pin, " set ", state

proc bitSend(bits: string) =
  traceDebug("bitSend: " & bits)
  BUS.add(bits)
  if not quietMode:
    echo "[BIT-SEND] ", bits

proc bitRecv() =
  traceDebug("bitRecv")
  if BUS.len > 0:
    let b = BUS[0]
    delete(BUS, 0)
    if not quietMode:
      echo "[BIT-RECV] ", b
  else:
    if not quietMode:
      echo "[BIT-RECV] empty"

proc memMap(target: string) =
  traceDebug("memMap: " & target)
  if not quietMode:
    echo "[MEM-MAP] ", target

proc memPush(target: string, value: string) =
  traceDebug("memPush: " & target & " <- " & value)
  if not quietMode:
    echo "[MEM-PUSH] ", target, " <- ", value

proc tranPulse(pin: int, width: string) =
  traceDebug("tranPulse: pin=" & $pin & " width=" & width)
  if not quietMode:
    echo "[TRAN-PULSE] pin ", pin, " width ", width

# --------------------------
# Generate Nim code + compile to binary release
# --------------------------
# --------------------------
# Code generation for "gpu ..." commands
# --------------------------
# Supported syntax:
#   gpu backend is "auto"          # "auto" | "cuda" (Vulkan alias) | "metal" | "cpu" | "vulkan"
#   gpu array X = [1, 2, 3, 4]
#   gpu add A, B -> C size 4       # "size N" optional, descriptive only
#   gpu sub A, B -> C
#   gpu mul A, B -> C
#   gpu div A, B -> C
#   gpu matmul A, B -> C m M k K n N
#   gpu relu X -> Y
#   gpu sigmoid X -> Y
#   gpu tanh X -> Y
#   gpu apflu X -> Y alpha A beta B
#   gpu softmax X -> Y rows R cols C
#   gpu layernorm X, GAMMA, BETA -> Y rows R cols C eps E
#   gpu layernorm_backward DY, X, GAMMA, BETA -> DX, DGAMMA, DBETA rows R cols C eps E
#   gpu embedding TABLE, INDICES -> Y vocab V dim D
#   gpu attention Q, K, V -> O, S_MATRIX B H S D scale SCALE
#   gpu attention_backward Q, K, V, S_MATRIX, DY -> DQ, DK, DV B H S D scale SCALE
proc genGpuLine*(line: string): seq[string] =
  traceDebug("Parsing GPU line: " & line)
  result = @[]
  let rest = line[3..^1].strip()

  if rest.startsWith("backend is"):
    let raw = rest["backend is".len..^1].strip()
    var val = stripQuotes(raw)
    if val == "auto":
      let backendOverride = getEnv("BYBYLANG_BACKEND").strip()
      if backendOverride.len > 0: val = backendOverride
    traceInfo("Setting GPU backend: " & val)
    result.add("gpuBackendSelected = parseBackend(\"" & val & "\")")

  elif rest.startsWith("array "):
    let body = rest["array ".len..^1].strip()
    let parts = body.split("=", 1)
    if parts.len == 2:
      let name = parts[0].strip()
      let valsRaw = parts[1].strip()
      traceDebug("Creating GPU array: " & name & " = " & valsRaw)
      result.add("var " & name & ": seq[float32] = @" & valsRaw & ".mapIt(it.float32)")
    else:
      traceWarn("Invalid gpu array syntax: " & line)
      result.add("# [WARN] invalid 'gpu array' syntax: " & line)

  else:
    let arrowParts = rest.split("->")
    if arrowParts.len == 2:
      let lhs = arrowParts[0].strip()
      let rhs = arrowParts[1].strip()
      let lhsWords = lhs.split()
      if lhsWords.len >= 1:
        let opName = lhsWords[0].toLowerAscii()
        let operandsStr = lhs[opName.len..^1].strip()
        let operands = operandsStr.split(",")
        traceDebug("GPU op: " & opName & " with " & $operands.len & " operands")
        
        # Special opName handling
        if opName == "layernorm_backward":
          if operands.len >= 4:
            let dy = operands[0].strip()
            let x = operands[1].strip()
            let gamma = operands[2].strip()
            let beta = operands[3].strip()
            let rhsParts = rhs.splitWhitespace()
            if rhsParts.len >= 9:
              let targetDx = rhsParts[0].replace(",", "").strip()
              let targetDgamma = rhsParts[1].replace(",", "").strip()
              let targetDbeta = rhsParts[2].replace(",", "").strip()
              var rows = "0"; var cols = "0"; var eps = "0.0"
              for i in 0..<rhsParts.len:
                if rhsParts[i] == "rows" and i+1 < rhsParts.len:
                  rows = rhsParts[i+1]
                elif rhsParts[i] == "cols" and i+1 < rhsParts.len:
                  cols = rhsParts[i+1]
                elif rhsParts[i] == "eps" and i+1 < rhsParts.len:
                  eps = rhsParts[i+1]
              if rows != "0" and cols != "0" and eps != "0.0":
                result.add("let lnBwd = gpuLayernormBackward(gpuBackendSelected, " & dy & ", " & x & ", " & gamma & ", " & beta & ", " & rows & ", " & cols & ", " & eps & ")")
                result.add("var " & targetDx & " = lnBwd.dx")
                result.add("var " & targetDgamma & " = lnBwd.dgamma")
                result.add("var " & targetDbeta & " = lnBwd.dbeta")
              else:
                traceWarn("Invalid layernorm_backward syntax: " & line)
                result.add("# [WARN] 'gpu layernorm_backward' syntax: gpu layernorm_backward DY, X, GAMMA, BETA -> DX, DGAMMA, DBETA rows R cols C eps E : " & line)
            else:
              traceWarn("Invalid layernorm_backward syntax: " & line)
              result.add("# [WARN] 'gpu layernorm_backward' syntax: gpu layernorm_backward DY, X, GAMMA, BETA -> DX, DGAMMA, DBETA rows R cols C eps E : " & line)
          else:
            traceWarn("layernorm_backward needs 4 operands: " & line)
            result.add("# [WARN] 'gpu layernorm_backward' needs 4 operands: " & line)
            
        elif opName == "attention":
          if operands.len >= 3:
            let q = operands[0].strip()
            let k = operands[1].strip()
            let v = operands[2].strip()
            let rhsParts = rhs.splitWhitespace()
            if rhsParts.len >= 11:
              let targetO = rhsParts[0].replace(",", "").strip()
              let targetS = rhsParts[1].replace(",", "").strip()
              var B = "0"; var H = "0"; var S = "0"; var D = "0"; var scale = "0.0"
              for i in 0..<rhsParts.len:
                if rhsParts[i] == "B" and i+1 < rhsParts.len:
                  B = rhsParts[i+1]
                elif rhsParts[i] == "H" and i+1 < rhsParts.len:
                  H = rhsParts[i+1]
                elif rhsParts[i] == "S" and i+1 < rhsParts.len:
                  S = rhsParts[i+1]
                elif rhsParts[i] == "D" and i+1 < rhsParts.len:
                  D = rhsParts[i+1]
                elif rhsParts[i] == "scale" and i+1 < rhsParts.len:
                  scale = rhsParts[i+1]
              if B != "0" and H != "0" and S != "0" and D != "0" and scale != "0.0":
                result.add("let attnRes = gpuAttentionFused(gpuBackendSelected, " & q & ", " & k & ", " & v & ", @[], " & B & ", " & H & ", " & S & ", " & D & ", " & scale & ")")
                result.add("var " & targetO & " = attnRes.o")
                result.add("var " & targetS & " = attnRes.s_matrix")
              else:
                traceWarn("Invalid attention syntax: " & line)
                result.add("# [WARN] 'gpu attention' syntax: gpu attention Q, K, V -> O, S_MATRIX B H S D scale SCALE : " & line)
            else:
              traceWarn("Invalid attention syntax: " & line)
              result.add("# [WARN] 'gpu attention' syntax: gpu attention Q, K, V -> O, S_MATRIX B H S D scale SCALE : " & line)
          else:
            traceWarn("attention needs 3 operands: " & line)
            result.add("# [WARN] 'gpu attention' needs 3 operands Q, K, V: " & line)
            
        elif opName == "attention_backward":
          if operands.len >= 5:
            let q = operands[0].strip()
            let k = operands[1].strip()
            let v = operands[2].strip()
            let s_matrix = operands[3].strip()
            let dy = operands[4].strip()
            let rhsParts = rhs.splitWhitespace()
            if rhsParts.len >= 11:
              let targetDq = rhsParts[0].replace(",", "").strip()
              let targetDk = rhsParts[1].replace(",", "").strip()
              let targetDv = rhsParts[2].replace(",", "").strip()
              var B = "0"; var H = "0"; var S = "0"; var D = "0"; var scale = "0.0"
              for i in 0..<rhsParts.len:
                if rhsParts[i] == "B" and i+1 < rhsParts.len:
                  B = rhsParts[i+1]
                elif rhsParts[i] == "H" and i+1 < rhsParts.len:
                  H = rhsParts[i+1]
                elif rhsParts[i] == "S" and i+1 < rhsParts.len:
                  S = rhsParts[i+1]
                elif rhsParts[i] == "D" and i+1 < rhsParts.len:
                  D = rhsParts[i+1]
                elif rhsParts[i] == "scale" and i+1 < rhsParts.len:
                  scale = rhsParts[i+1]
              if B != "0" and H != "0" and S != "0" and D != "0" and scale != "0.0":
                result.add("let attnBwd = gpuAttentionFusedBackward(gpuBackendSelected, " & q & ", " & k & ", " & v & ", " & s_matrix & ", " & dy & ", " & B & ", " & H & ", " & S & ", " & D & ", " & scale & ")")
                result.add("var " & targetDq & " = attnBwd.dq")
                result.add("var " & targetDk & " = attnBwd.dk")
                result.add("var " & targetDv & " = attnBwd.dv")
              else:
                traceWarn("Invalid attention_backward syntax: " & line)
                result.add("# [WARN] 'gpu attention_backward' syntax: gpu attention_backward Q, K, V, S_MATRIX, DY -> DQ, DK, DV B H S D scale SCALE : " & line)
            else:
              traceWarn("Invalid attention_backward syntax: " & line)
              result.add("# [WARN] 'gpu attention_backward' syntax: gpu attention_backward Q, K, V, S_MATRIX, DY -> DQ, DK, DV B H S D scale SCALE : " & line)
          else:
            traceWarn("attention_backward needs 5 operands: " & line)
            result.add("# [WARN] 'gpu attention_backward' needs 5 operands: " & line)
            
        elif opName == "apflu":
          if operands.len >= 1:
            let x = operands[0].strip()
            let rhsParts = rhs.splitWhitespace()
            if rhsParts.len >= 5 and rhsParts[1] == "alpha" and rhsParts[3] == "beta":
              let target = rhsParts[0]
              let alpha = rhsParts[2]
              let beta = rhsParts[4]
              result.add("var " & target & " = gpuApflu(gpuBackendSelected, " & x & ", " & alpha & ", " & beta & ")")
            else:
              result.add("var " & rhsParts[0] & " = gpuApflu(gpuBackendSelected, " & x & ", 0.1'f32, 1.0'f32)")
          else:
            traceWarn("apflu needs 1 operand: " & line)
            result.add("# [WARN] 'gpu apflu' needs 1 operands: " & line)

        elif opName == "permute":
          # NEW: gpu permute SRC -> Y newStrides NS oldStrides OS dimsMap DM n N
          # NS/OS/DM are NAMES of existing seq[int32] (or seq[int]) variables, N is a scalar.
          if operands.len >= 1:
            let src = operands[0].strip()
            let rhsParts = rhs.splitWhitespace()
            let target = rhsParts[0]
            let ns = findNamedParam(rhsParts, "newStrides", "")
            let os = findNamedParam(rhsParts, "oldStrides", "")
            let dm = findNamedParam(rhsParts, "dimsMap", "")
            let n = findNamedParam(rhsParts, "n", "0")
            if ns.len > 0 and os.len > 0 and dm.len > 0:
              result.add("var " & target & " = gpuPermute(gpuBackendSelected, " & src & ", " & ns & ".mapIt(int32(it)), " & os & ".mapIt(int32(it)), " & dm & ".mapIt(int32(it)), " & n & ")")
            else:
              traceWarn("Invalid permute syntax: " & line)
              result.add("# [WARN] 'gpu permute' syntax: gpu permute SRC -> Y newStrides NS oldStrides OS dimsMap DM n N : " & line)
          else:
            traceWarn("permute needs 1 operand: " & line)
            result.add("# [WARN] 'gpu permute' needs 1 operands: " & line)

        elif opName == "split_slice":
          # THEM MOI: gpu split_slice SRC -> Y dimTotal DT thisSize TS offset OFF inner IN n N
          if operands.len >= 1:
            let src = operands[0].strip()
            let rhsParts = rhs.splitWhitespace()
            let target = rhsParts[0]
            let dimTotal = findNamedParam(rhsParts, "dimTotal", "0")
            let thisSize = findNamedParam(rhsParts, "thisSize", "0")
            let offset = findNamedParam(rhsParts, "offset", "0")
            let inner = findNamedParam(rhsParts, "inner", "0")
            let n = findNamedParam(rhsParts, "n", "0")
            result.add("var " & target & " = gpuSplitSlice(gpuBackendSelected, " & src & ", " & dimTotal & ", " & thisSize & ", " & offset & ", " & inner & ", " & n & ")")
          else:
            traceWarn("split_slice needs 1 operand: " & line)
            result.add("# [WARN] 'gpu split_slice' needs 1 operands: " & line)

        elif opName == "concat":
          # THEM MOI: gpu concat FLATSRC -> Y srcBase SB srcDimSize SDS srcDimOffset SDO
          #           numTensors NT totalDim TD inner IN
          # SB/SDS/SDO are the NAMES of existing seq[int32] (or seq[int]) variables.
          if operands.len >= 1:
            let flatSrc = operands[0].strip()
            let rhsParts = rhs.splitWhitespace()
            let target = rhsParts[0]
            let sb = findNamedParam(rhsParts, "srcBase", "")
            let sds = findNamedParam(rhsParts, "srcDimSize", "")
            let sdo = findNamedParam(rhsParts, "srcDimOffset", "")
            let numT = findNamedParam(rhsParts, "numTensors", "0")
            let totalDim = findNamedParam(rhsParts, "totalDim", "0")
            let inner = findNamedParam(rhsParts, "inner", "0")
            if sb.len > 0 and sds.len > 0 and sdo.len > 0:
              result.add("var " & target & " = gpuConcat(gpuBackendSelected, " & flatSrc & ", " & sb & ".mapIt(int32(it)), " & sds & ".mapIt(int32(it)), " & sdo & ".mapIt(int32(it)), " & numT & ", " & totalDim & ", " & inner & ")")
            else:
              traceWarn("Invalid concat syntax: " & line)
              result.add("# [WARN] 'gpu concat' syntax: gpu concat FLATSRC -> Y srcBase SB srcDimSize SDS srcDimOffset SDO numTensors NT totalDim TD inner IN : " & line)
          else:
            traceWarn("concat needs 1 operand: " & line)
            result.add("# [WARN] 'gpu concat' needs 1 operands: " & line)

        elif opName == "argmax_index":
          # THEM MOI: gpu argmax_index X -> VAL, IDX outer O dimSize D inner I
          if operands.len >= 1:
            let x = operands[0].strip()
            let rhsMain = rhs.split("outer")[0]
            let rhsTargets = rhsMain.splitWhitespace().mapIt(it.replace(",", "").strip())
            let rhsParts = rhs.splitWhitespace()
            if rhsTargets.len >= 2:
              let targetVal = rhsTargets[0]
              let targetIdx = rhsTargets[1]
              let outer = findNamedParam(rhsParts, "outer", "0")
              let dimSize = findNamedParam(rhsParts, "dimSize", "0")
              let inner = findNamedParam(rhsParts, "inner", "0")
              result.add("let amRes = gpuArgmaxIndex(gpuBackendSelected, " & x & ", " & outer & ", " & dimSize & ", " & inner & ")")
              result.add("var " & targetVal & " = amRes.val")
              result.add("var " & targetIdx & " = amRes.idx")
            else:
              traceWarn("Invalid argmax_index syntax: " & line)
              result.add("# [WARN] 'gpu argmax_index' syntax: gpu argmax_index X -> VAL, IDX outer O dimSize D inner I : " & line)
          else:
            traceWarn("argmax_index needs 1 operand: " & line)
            result.add("# [WARN] 'gpu argmax_index' needs 1 operands: " & line)

        elif opName == "dropout_mask":
          # NEW: gpu dropout_mask -> Y n N p P seed S  (NO input
          # operand - this op GENERATES a new mask, it does not transform a tensor,
          # so "lhs" is only the op name, no operand before "->")
          let rhsParts = rhs.splitWhitespace()
          let target = rhsParts[0]
          let n = findNamedParam(rhsParts, "n", "0")
          let p = findNamedParam(rhsParts, "p", "0.5")
          let seed = findNamedParam(rhsParts, "seed", "42")
          result.add("var " & target & " = gpuDropoutMask(gpuBackendSelected, " & n & ", " & p & "'f32, uint32(" & seed & "))")

        else:
          if operands.len >= 2:
            let a = operands[0].strip()
            let b = operands[1].strip()

            if opName == "matmul":
              let rhsParts = rhs.splitWhitespace()
              if rhsParts.len >= 7 and rhsParts[1] == "m" and rhsParts[3] == "k" and rhsParts[5] == "n":
                let target = rhsParts[0]
                let m = rhsParts[2]; let k = rhsParts[4]; let n = rhsParts[6]
                result.add("var " & target & " = gpuMatmul(gpuBackendSelected, " & a & ", " & b & ", " & m & ", " & k & ", " & n & ")")
              else:
                traceWarn("Invalid matmul syntax: " & line)
                result.add("# [WARN] 'gpu matmul' syntax: gpu matmul A, B -> C m M k K n N : " & line)
            elif opName == "layernorm":
              if operands.len >= 3:
                let gamma = operands[1].strip()
                let beta = operands[2].strip()
                let rhsParts = rhs.splitWhitespace()
                if rhsParts.len >= 7 and rhsParts[1] == "rows" and rhsParts[3] == "cols" and rhsParts[5] == "eps":
                  let target = rhsParts[0]
                  let rows = rhsParts[2]; let cols = rhsParts[4]; let eps = rhsParts[6]
                  result.add("var " & target & " = gpuLayernorm(gpuBackendSelected, " & a & ", " & gamma & ", " & beta & ", " & rows & ", " & cols & ", " & eps & ")")
                else:
                  traceWarn("Invalid layernorm syntax: " & line)
                  result.add("# [WARN] 'gpu layernorm' syntax: gpu layernorm X, GAMMA, BETA -> Y rows R cols C eps E : " & line)
              else:
                traceWarn("layernorm needs 3 operands: " & line)
                result.add("# [WARN] 'gpu layernorm' needs 3 operands: " & line)
            elif opName == "embedding":
              let rhsParts = rhs.splitWhitespace()
              if rhsParts.len >= 5 and rhsParts[1] == "vocab" and rhsParts[3] == "dim":
                let target = rhsParts[0]
                let vocab = rhsParts[2]; let dim = rhsParts[4]
                result.add("var " & target & " = gpuEmbeddingLookup(gpuBackendSelected, " & a & ", " & b & ".mapIt(int32(it)), " & vocab & ", " & dim & ")")
              else:
                traceWarn("Invalid embedding syntax: " & line)
                result.add("# [WARN] 'gpu embedding' syntax: gpu embedding TABLE, INDICES -> Y vocab V dim D : " & line)
            elif opName == "gather":
              # NEW: gpu gather SRC, IDX -> Y  (IDX is a seq[int32]
              # or seq[int] variable - .mapIt(int32(it)) is safe for both)
              let target = rhs.splitWhitespace()[0]
              result.add("var " & target & " = gpuGather(gpuBackendSelected, " & a & ", " & b & ".mapIt(int32(it)))")
            elif opName == "compare":
              # THEM MOI: gpu compare A, B -> Y op 0   (op: 0=gt 1=lt 2=eq 3=ge 4=le)
              let rhsParts = rhs.splitWhitespace()
              let target = rhsParts[0]
              let cmpOp = findNamedParam(rhsParts, "op", "2")
              result.add("var " & target & " = gpuCompare(gpuBackendSelected, " & a & ", " & b & ", " & cmpOp & ")")
            elif opName == "fused_add_act":
              # THEM MOI: gpu fused_add_act A, B -> Y act "relu"
              let rhsParts = rhs.splitWhitespace()
              let target = rhsParts[0]
              let act = stripQuotes(findNamedParam(rhsParts, "act", "\"relu\""))
              result.add("var " & target & " = gpuFusedAddAct(gpuBackendSelected, " & a & ", " & b & ", \"" & act & "\")")
            elif opName == "apflu_backward":
              # THEM MOI: gpu apflu_backward X, DY -> Y alpha 0.1 beta 0.1
              let rhsParts = rhs.splitWhitespace()
              let target = rhsParts[0]
              let alpha = findNamedParam(rhsParts, "alpha", "0.1")
              let beta = findNamedParam(rhsParts, "beta", "0.1")
              result.add("var " & target & " = gpuApfluBackward(gpuBackendSelected, " & a & ", " & b & ", " & alpha & "'f32, " & beta & "'f32)")
            else:
              var target = rhs
              if "size" in rhs:
                target = rhs.split("size")[0].strip()
              result.add("var " & target & " = gpuOp(\"" & opName & "\", gpuBackendSelected, " & a & ", " & b & ")")
          elif operands.len == 1:
            let x = operands[0].strip()
            case opName
            of "relu", "sigmoid", "tanh":
              let target = rhs.splitWhitespace()[0]
              let fn = if opName == "relu": "gpuRelu"
                       elif opName == "sigmoid": "gpuSigmoid"
                       else: "gpuTanh"
              result.add("var " & target & " = " & fn & "(gpuBackendSelected, " & x & ")")
            of "sqrt", "exp", "log":
              # THEM MOI: gpu sqrt X -> Y / gpu exp X -> Y / gpu log X -> Y
              let target = rhs.splitWhitespace()[0]
              let fn = if opName == "sqrt": "gpuSqrt"
                       elif opName == "exp": "gpuExp"
                       else: "gpuLog"
              result.add("var " & target & " = " & fn & "(gpuBackendSelected, " & x & ")")
            of "pow":
              # THEM MOI: gpu pow X -> Y p 2.0
              let rhsParts = rhs.splitWhitespace()
              let target = rhsParts[0]
              let p = findNamedParam(rhsParts, "p", "2.0")
              result.add("var " & target & " = gpuPow(gpuBackendSelected, " & x & ", " & p & "'f32)")
            of "softmax":
              let rhsParts = rhs.splitWhitespace()
              if rhsParts.len >= 5 and rhsParts[1] == "rows" and rhsParts[3] == "cols":
                let target = rhsParts[0]
                let rows = rhsParts[2]; let cols = rhsParts[4]
                result.add("var " & target & " = gpuSoftmax(gpuBackendSelected, " & x & ", " & rows & ", " & cols & ")")
              else:
                traceWarn("Invalid softmax syntax: " & line)
                result.add("# [WARN] 'gpu softmax' syntax: gpu softmax X -> Y rows R cols C : " & line)
            else:
              traceWarn("Unknown unary op: " & opName)
              result.add("# [WARN] 'gpu " & opName & "' needs 2 operands: " & line)
          else:
            traceWarn("No operands for op: " & opName)
            result.add("# [WARN] 'gpu " & opName & "' needs operands: " & line)
      else:
        traceWarn("No words in LHS: " & lhs)
        result.add("# [WARN] invalid gpu syntax: " & line)
    else:
      traceWarn("No arrow found: " & rest)
      result.add("# [WARN] invalid gpu syntax (missing '->'): " & line)
# --------------------------
# Recursive code generation for a block of statements
# --------------------------
# Nests to ANY DEPTH: if in if, while in if, for in while,
# a gpu call in an if in a for, etc. Each statement that opens a new block
# (if/elif/else/while/for) calls this function again (recursively) for its body,
# instead of duplicating the handling code per level like the old version.
proc genBlock(tokens: seq[Token], idx: var int, indent: string,
              funcNames: seq[string], localVars: var seq[string]): seq[string]

proc genRawBlock(tokens: seq[Token], idx: var int, indent: string,
                 blockIndent: int): seq[string] =
  ## A type section is plain Nim: emit its header and every deeper line
  ## unchanged, keeping their indentation relative to the first body line.
  result = @[indent & tokens[idx].text]
  idx.inc
  var bodyIndent = -1
  while idx < tokens.len:
    let j = nextCode(tokens, idx)
    if j >= tokens.len or tokens[j].indent <= blockIndent: break
    if bodyIndent < 0: bodyIndent = tokens[j].indent
    let t = tokens[j]
    result.add(indent & "  " & " ".repeat(max(0, t.indent - bodyIndent)) & t.text)
    idx = j + 1

# Shared by if/elif/else/while/for: emits the header line, then either recurses into
# the body (when the next line is indented deeper) or emits "discard" (empty block).
proc genHeaderedBlock(tokens: seq[Token], idx: var int, indent: string, headerLine: string,
                       blockIndent: int, funcNames: seq[string], localVars: var seq[string]): seq[string] =
  result = @[indent & headerLine]
  idx.inc
  let body = nextCode(tokens, idx)
  if body < tokens.len and tokens[body].indent > blockIndent:
    result.add(genBlock(tokens, idx, indent & "  ", funcNames, localVars))
  else:
    result.add(indent & "  discard")
# --------------------------
# Code generation for a "session:" block - uses the Session API (sessionBegin/
# sessionUpload/session<Op>/sessionEnd/sessionRead/sessionFree) of
# gpubackend.nim to batch several "gpu ..." commands into one GPU session.
# --------------------------
proc genSessionBlock(tokens: seq[Token], idx: var int, indent: string,
                      blockIndent: int, localVars: var seq[string]): seq[string] =
  result = @[]
  idx.inc  # skip the "session:" line

  var bodyLines: seq[string] = @[]
  if idx < tokens.len and tokens[idx].indent > blockIndent:
    let sessionIndent = tokens[idx].indent
    while idx < tokens.len and tokens[idx].indent >= sessionIndent:
      let t = tokens[idx]
      if t.sym notin ["empty", "comment"]:
        bodyLines.add(t.text.strip())
      idx.inc
  else:
    traceWarn("Empty 'session:' block")
    result.add(indent & "discard")
    return

  traceInfo("Session block with " & $bodyLines.len & " lines")

  var uploaded = initTable[string, string]()      # variable name (already on the CPU) -> handle name
  var produced = initTable[string, string]()      # variable name (created INSIDE the session, not on the CPU yet) -> handle name
  var producedSize = initTable[string, string]()  # variable name (produced) -> size expression (no .len yet)
  var handleDecls: seq[string] = @[]
  var opLines: seq[string] = @[]
  var resultHandles: seq[tuple[varName: string, handleVar: string, size: string]] = @[]
  var extraFreeHandles: seq[string] = @[]           # intermediate handles (e.g. the add buffer before the activation in fused_add_act) - not in resultHandles, so freed separately
  var passthrough: seq[string] = @[]               # lines run normally, outside the session
  var sessionSourceLines: seq[string] = @[]         # the "gpu ..." lines batched in the session (for the fallback)

  var hcount = 0
  proc newHandle(): string =
    hcount.inc
    return "sh" & $hcount

  # If varName was just produced by an earlier command IN THE SAME session (still
  # on the GPU, not read back to the CPU), reuse that GPU handle - no
  # re-upload from the Nim variable (which does not exist yet at this point).
  # Otherwise upload the existing CPU variable (once, cached).
  proc ensureUploaded(varName: string): string =
    if varName in produced:
      return produced[varName]
    if varName in uploaded:
      return uploaded[varName]
    let h = newHandle()
    handleDecls.add("    var " & h & " = sessionUpload(gpuBackendSelected, " & varName & ")")
    uploaded[varName] = h
    return h

  # Size (element count) of varName. If varName was just produced INSIDE the
  # session (no real seq[float32] on the CPU yet), reuse the size
  # expression known when it was produced instead of "varName.len" (the variable
  # does not exist at this step).
  proc sizeOf(varName: string): string =
    if varName in producedSize:
      return producedSize[varName]
    return varName & ".len"

  for line in bodyLines:
    if line.len == 0: continue
    if not line.startsWith("gpu "):
      passthrough.add(line)
      continue

    let rest = line[3..^1].strip()
    let arrowParts = rest.split("->")
    if arrowParts.len != 2:
      passthrough.add(line)   # e.g. "gpu backend is ..." / "gpu array ..."
      continue

    let lhs = arrowParts[0].strip()
    let rhs = arrowParts[1].strip()
    let lhsWords = lhs.split()
    if lhsWords.len < 1:
      passthrough.add(line)
      continue

    let opName = lhsWords[0].toLowerAscii()
    let operandsStr = lhs[opName.len..^1].strip()
    let operands = operandsStr.split(",").mapIt(it.strip())

    case opName
    of "add", "sub", "mul", "div":
      if operands.len >= 2:
        var target = rhs
        var sizeExpr = sizeOf(operands[0])
        if "size" in rhs:
          let parts2 = rhs.split("size")
          target = parts2[0].strip()
          sizeExpr = parts2[1].strip()
        let ha = ensureUploaded(operands[0])
        let hb = ensureUploaded(operands[1])
        let hc = newHandle()
        handleDecls.add("    var " & hc & ": SessionHandle")
        opLines.add("    discard sessionVecOp(gpuBackendSelected, \"" & opName & "\", " &
                    ha & ", " & hb & ", " & hc & ", " & sizeExpr & ")")
        resultHandles.add((target, hc, sizeExpr))
        produced[target] = hc
        producedSize[target] = sizeExpr
        sessionSourceLines.add(line)
      else:
        passthrough.add(line)

    of "relu", "sigmoid", "tanh":
      if operands.len >= 1 and rhs.splitWhitespace().len >= 1:
        let x = operands[0]
        let target = rhs.splitWhitespace()[0]
        let ha = ensureUploaded(x)
        let hc = newHandle()
        handleDecls.add("    var " & hc & ": SessionHandle")
        let xsize = sizeOf(x)
        opLines.add("    discard sessionActivation(gpuBackendSelected, \"" & opName & "\", " &
                    ha & ", " & hc & ", " & xsize & ")")
        resultHandles.add((target, hc, xsize))
        produced[target] = hc
        producedSize[target] = xsize
        sessionSourceLines.add(line)
      else:
        passthrough.add(line)

    of "softmax":
      let rhsParts = rhs.splitWhitespace()
      if operands.len >= 1 and rhsParts.len >= 5 and rhsParts[1] == "rows" and rhsParts[3] == "cols":
        let x = operands[0]
        let target = rhsParts[0]
        let rows = rhsParts[2]; let cols = rhsParts[4]
        let ha = ensureUploaded(x)
        let hc = newHandle()
        handleDecls.add("    var " & hc & ": SessionHandle")
        opLines.add("    discard sessionSoftmax(gpuBackendSelected, " & ha & ", " & hc &
                    ", " & rows & ", " & cols & ")")
        resultHandles.add((target, hc, rows & " * " & cols))
        produced[target] = hc
        producedSize[target] = rows & " * " & cols
        sessionSourceLines.add(line)
      else:
        passthrough.add(line)

    of "matmul":
      let rhsParts = rhs.splitWhitespace()
      if operands.len >= 2 and rhsParts.len >= 7 and rhsParts[1] == "m" and
         rhsParts[3] == "k" and rhsParts[5] == "n":
        let ha = ensureUploaded(operands[0])
        let hb = ensureUploaded(operands[1])
        let target = rhsParts[0]
        let m = rhsParts[2]; let k = rhsParts[4]; let n = rhsParts[6]
        let hc = newHandle()
        handleDecls.add("    var " & hc & ": SessionHandle")
        opLines.add("    discard sessionMatmul(gpuBackendSelected, " & ha & ", " & hb &
                    ", " & hc & ", " & m & ", " & k & ", " & n & ")")
        resultHandles.add((target, hc, m & " * " & n))
        produced[target] = hc
        producedSize[target] = m & " * " & n
        sessionSourceLines.add(line)
      else:
        passthrough.add(line)

    of "layernorm":
      let rhsParts = rhs.splitWhitespace()
      if operands.len >= 3 and rhsParts.len >= 5 and rhsParts[1] == "rows" and rhsParts[3] == "cols":
        let hx = ensureUploaded(operands[0])
        let hg = ensureUploaded(operands[1])
        let hb = ensureUploaded(operands[2])
        let target = rhsParts[0]
        let rows = rhsParts[2]; let cols = rhsParts[4]
        let hc = newHandle()
        handleDecls.add("    var " & hc & ": SessionHandle")
        opLines.add("    discard sessionLayernormEnc(gpuBackendSelected, " & hx & ", " & hg &
                    ", " & hb & ", " & hc & ", " & rows & ", " & cols & ")")
        resultHandles.add((target, hc, rows & " * " & cols))
        produced[target] = hc
        producedSize[target] = rows & " * " & cols
        sessionSourceLines.add(line)
      else:
        passthrough.add(line)

    of "rope":
      # gpu rope X, COS, SIN -> Y seqlen T heads H dim D
      let rhsParts = rhs.splitWhitespace()
      if operands.len >= 3 and rhsParts.len >= 7 and rhsParts[1] == "seqlen" and
         rhsParts[3] == "heads" and rhsParts[5] == "dim":
        let hx = ensureUploaded(operands[0])
        let hcos = ensureUploaded(operands[1])
        let hsin = ensureUploaded(operands[2])
        let target = rhsParts[0]
        let seqlen = rhsParts[2]; let heads = rhsParts[4]; let dim = rhsParts[6]
        opLines.add("    discard sessionRope(gpuBackendSelected, " & hx & ", " & hcos &
                    ", " & hsin & ", " & seqlen & ", " & heads & ", " & dim & ")")
        # sessionRope rotates handle X in place -> read hx back as the result
        resultHandles.add((target, hx, operands[0] & ".len"))
        produced[target] = hx
        producedSize[target] = sizeOf(operands[0])
        sessionSourceLines.add(line)
      else:
        traceWarn("'gpu rope' syntax: gpu rope X, COS, SIN -> Y seqlen T heads H dim D")
        passthrough.add(line)

    of "fused_add_act":
      # gpu fused_add_act A, B -> Y act "relu"
      # Encoded DIRECTLY into the enclosing session with sessionVecOp("add") +
      # sessionActivation(act) - the SAME 2 real kernels gpuFusedAddAct()
      # (gpubackend.nim) uses, but here they share the sessionBegin/End of
      # the other "gpu ..." commands in the same "session:" block instead of
      # opening a nested session (a separate gpuFusedAddAct call does its own
      # sessionBegin/End - still correct, just not batched).
      let rhsParts = rhs.splitWhitespace()
      if operands.len >= 2 and rhsParts.len >= 1:
        let target = rhsParts[0]
        let act = stripQuotes(findNamedParam(rhsParts, "act", "\"relu\""))
        let ha = ensureUploaded(operands[0])
        let hb = ensureUploaded(operands[1])
        let hc = newHandle()
        let sizeExpr = sizeOf(operands[0])
        handleDecls.add("    var " & hc & ": SessionHandle")
        opLines.add("    discard sessionVecOp(gpuBackendSelected, \"add\", " &
                    ha & ", " & hb & ", " & hc & ", " & sizeExpr & ")")
        if act == "none":
          resultHandles.add((target, hc, sizeExpr))
          produced[target] = hc
        else:
          let hy = newHandle()
          handleDecls.add("    var " & hy & ": SessionHandle")
          opLines.add("    discard sessionActivation(gpuBackendSelected, \"" & act & "\", " &
                      hc & ", " & hy & ", " & sizeExpr & ")")
          resultHandles.add((target, hy, sizeExpr))
          produced[target] = hy
          extraFreeHandles.add(hc)
        producedSize[target] = sizeExpr
        sessionSourceLines.add(line)
      else:
        passthrough.add(line)

    of "sqrt", "exp", "log":
      if operands.len >= 1 and rhs.splitWhitespace().len >= 1:
        let x = operands[0]
        let target = rhs.splitWhitespace()[0]
        let ha = ensureUploaded(x)
        let hc = newHandle()
        handleDecls.add("    var " & hc & ": SessionHandle")
        let xsize = sizeOf(x)
        opLines.add("    discard sessionActivation(gpuBackendSelected, \"" & opName & "\", " &
                    ha & ", " & hc & ", " & xsize & ")")
        resultHandles.add((target, hc, xsize))
        produced[target] = hc
        producedSize[target] = xsize
        sessionSourceLines.add(line)
      else:
        passthrough.add(line)

    of "pow":
      # gpu pow A -> Y p 2.0 - opCode 7, p must be passed through sessionActivation.
      let rhsParts = rhs.splitWhitespace()
      if operands.len >= 1 and rhsParts.len >= 1:
        let x = operands[0]
        let target = rhsParts[0]
        let p = findNamedParam(rhsParts, "p", "2.0")
        let ha = ensureUploaded(x)
        let hc = newHandle()
        handleDecls.add("    var " & hc & ": SessionHandle")
        let xsize = sizeOf(x)
        opLines.add("    discard sessionActivation(gpuBackendSelected, \"pow\", " &
                    ha & ", " & hc & ", " & xsize & ", " & p & "'f32)")
        resultHandles.add((target, hc, xsize))
        produced[target] = hc
        producedSize[target] = xsize
        sessionSourceLines.add(line)
      else:
        passthrough.add(line)

    of "apflu":
      # gpu apflu A -> Y alpha 0.1 beta 1.0 - Metal supports ONLY
      # alpha=0.1/beta=1.0 inside a session (see sessionApfluEnc); if the backend
      # is Metal with other alpha/beta, sessionApfluEnc returns false
      # -> the line falls back to immediate (gpuApflu is correct for every
      # alpha/beta because it is a more general CPU/kernel function).
      let rhsParts = rhs.splitWhitespace()
      if operands.len >= 1 and rhsParts.len >= 1:
        let x = operands[0]
        let target = rhsParts[0]
        let alpha = findNamedParam(rhsParts, "alpha", "0.1")
        let beta = findNamedParam(rhsParts, "beta", "1.0")
        let ha = ensureUploaded(x)
        let hc = newHandle()
        handleDecls.add("    var " & hc & ": SessionHandle")
        let xsize = sizeOf(x)
        opLines.add("    discard sessionApfluEnc(gpuBackendSelected, " & ha & ", " & hc &
                    ", " & xsize & ", " & alpha & "'f32, " & beta & "'f32)")
        resultHandles.add((target, hc, xsize))
        produced[target] = hc
        producedSize[target] = xsize
        sessionSourceLines.add(line)
      else:
        passthrough.add(line)

    of "apflu_backward":
      let rhsParts = rhs.splitWhitespace()
      if operands.len >= 2 and rhsParts.len >= 1:
        let x = operands[0]; let dy = operands[1]
        let target = rhsParts[0]
        let alpha = findNamedParam(rhsParts, "alpha", "0.1")
        let beta = findNamedParam(rhsParts, "beta", "0.1")
        let hx = ensureUploaded(x)
        let hdy = ensureUploaded(dy)
        let hc = newHandle()
        handleDecls.add("    var " & hc & ": SessionHandle")
        let xsize = sizeOf(x)
        opLines.add("    discard sessionApfluBackwardEnc(gpuBackendSelected, " & hx & ", " & hdy &
                    ", " & hc & ", " & xsize & ", " & alpha & "'f32, " & beta & "'f32)")
        resultHandles.add((target, hc, xsize))
        produced[target] = hc
        producedSize[target] = xsize
        sessionSourceLines.add(line)
      else:
        passthrough.add(line)

    of "gather":
      # IDX must be an int array (seq[int]/seq[int32]) uploaded through
      # sessionUploadIndices - NOT ensureUploaded (only for
      # seq[float32]). Not cached (indices are usually small and rarely reused).
      if operands.len >= 2 and rhs.splitWhitespace().len >= 1:
        let src = operands[0]; let idx = operands[1]
        let target = rhs.splitWhitespace()[0]
        let hsrc = ensureUploaded(src)
        let hidx = newHandle()
        handleDecls.add("    var " & hidx & " = sessionUploadIndices(gpuBackendSelected, " &
                    idx & ".mapIt(int32(it)))")
        let hc = newHandle()
        handleDecls.add("    var " & hc & ": SessionHandle")
        let idxSize = idx & ".len"
        opLines.add("    discard sessionGatherEnc(gpuBackendSelected, " & hsrc & ", " & hidx &
                    ", " & hc & ", " & idxSize & ")")
        resultHandles.add((target, hc, idxSize))
        produced[target] = hc
        producedSize[target] = idxSize
        extraFreeHandles.add(hidx)
        sessionSourceLines.add(line)
      else:
        passthrough.add(line)

    of "compare":
      let rhsParts = rhs.splitWhitespace()
      if operands.len >= 2 and rhsParts.len >= 1:
        let a = operands[0]; let b = operands[1]
        let target = rhsParts[0]
        let cmpOp = findNamedParam(rhsParts, "op", "2")
        let ha = ensureUploaded(a)
        let hb = ensureUploaded(b)
        let hc = newHandle()
        handleDecls.add("    var " & hc & ": SessionHandle")
        let sizeExpr = sizeOf(a)
        opLines.add("    discard sessionCompareEnc(gpuBackendSelected, " & ha & ", " & hb &
                    ", " & hc & ", " & cmpOp & ", " & sizeExpr & ")")
        resultHandles.add((target, hc, sizeExpr))
        produced[target] = hc
        producedSize[target] = sizeExpr
        sessionSourceLines.add(line)
      else:
        passthrough.add(line)

    of "dropout_mask":
      # NO input operand (the op generates a new mask) - "rhs" holds both the target
      # and the n/p/seed parameters.
      let rhsParts = rhs.splitWhitespace()
      if rhsParts.len >= 1:
        let target = rhsParts[0]
        let n = findNamedParam(rhsParts, "n", "0")
        let p = findNamedParam(rhsParts, "p", "0.5")
        let seed = findNamedParam(rhsParts, "seed", "42")
        let hc = newHandle()
        handleDecls.add("    var " & hc & ": SessionHandle")
        opLines.add("    discard sessionDropoutMaskEnc(gpuBackendSelected, " & hc & ", " &
                    n & ", " & p & "'f32, uint32(" & seed & "))")
        resultHandles.add((target, hc, n))
        produced[target] = hc
        producedSize[target] = n
        sessionSourceLines.add(line)
      else:
        passthrough.add(line)

    of "split_slice":
      if operands.len >= 1 and rhs.splitWhitespace().len >= 1:
        let src = operands[0]
        let rhsParts = rhs.splitWhitespace()
        let target = rhsParts[0]
        let dimTotal = findNamedParam(rhsParts, "dimTotal", "0")
        let thisSize = findNamedParam(rhsParts, "thisSize", "0")
        let offset = findNamedParam(rhsParts, "offset", "0")
        let inner = findNamedParam(rhsParts, "inner", "0")
        let n = findNamedParam(rhsParts, "n", "0")
        let hsrc = ensureUploaded(src)
        let hc = newHandle()
        handleDecls.add("    var " & hc & ": SessionHandle")
        opLines.add("    discard sessionSplitSliceEnc(gpuBackendSelected, " & hsrc & ", " & hc &
                    ", " & dimTotal & ", " & thisSize & ", " & offset & ", " & inner & ", " & n & ")")
        resultHandles.add((target, hc, n))
        produced[target] = hc
        producedSize[target] = n
        sessionSourceLines.add(line)
      else:
        passthrough.add(line)

    else:
      traceWarn("GPU op '" & opName & "' has no session variant - using immediate")
      passthrough.add(line)

  # --- Generate the Nim block: try the session, fall back to immediate if it cannot open ---
  # The result variables MUST be declared "var" OUTSIDE the block (Nim's "block:"
  # opens its own scope - a declaration inside would vanish when leaving the
  # block); inside, they are only assigned (no "var").
  for rh in resultHandles:
    if rh.varName notin localVars:
      result.add(indent & "var " & rh.varName & ": seq[float32] = @[]")
      localVars.add(rh.varName)

  result.add(indent & "block:")
  if opLines.len > 0:
    result.add(indent & "  var sessOk = sessionBegin(gpuBackendSelected)")
    result.add(indent & "  if sessOk:")
    for d in handleDecls: result.add(indent & d)
    for o in opLines: result.add(indent & o)
    result.add(indent & "    discard sessionEnd(gpuBackendSelected)")
    for rh in resultHandles:
      result.add(indent & "    " & rh.varName & " = sessionRead(gpuBackendSelected, " &
                 rh.handleVar & ", " & rh.size & ")")
    var freed: seq[string] = @[]
    for rh in resultHandles:
      if rh.handleVar notin freed:
        result.add(indent & "    sessionFree(gpuBackendSelected, " & rh.handleVar & ")")
        freed.add(rh.handleVar)
    for h in extraFreeHandles:
      if h notin freed:
        result.add(indent & "    sessionFree(gpuBackendSelected, " & h & ")")
        freed.add(h)
    for _, h in uploaded:
      if h notin freed:
        result.add(indent & "    sessionFree(gpuBackendSelected, " & h & ")")
        freed.add(h)
    result.add(indent & "  else:")
    result.add(indent & "    echo \"[WARN] session could not be started, falling back to immediate\"")
    # sessionSourceLines and resultHandles are filled in parallel (1-1, same
    # order) by the loop above, so they pair up by index.
    for i in 0..<sessionSourceLines.len:
      let l = sessionSourceLines[i]
      let rh = resultHandles[i]
      if l.startsWith("gpu rope"):
        # rope has NO immediate variant (gpubackend.nim only has sessionRope) -
        # it cannot run outside a session, so the fallback keeps the input.
        result.add(indent & "    echo \"[WARN] gpu rope: the backend has no session support, skipping RoPE (input unchanged)\"")
        let firstOperand = l.split("gpu rope")[1].split("->")[0].split(",")[0].strip()
        result.add(indent & "    " & rh.varName & " = " & firstOperand)
      else:
        for gl in genGpuLine(l):
          var fixed = gl
          let varDecl = "var " & rh.varName & " "
          if fixed.startsWith(varDecl):
            fixed = rh.varName & fixed[("var " & rh.varName).len..^1]
          result.add(indent & "    " & fixed)
  else:
    result.add(indent & "  discard  # no gpu command in this block could be batched in a session")

  # --- The other lines (not session-batchable gpu commands) run normally ---
  for l in passthrough:
    if l.startsWith("gpu "):
      for gl in genGpuLine(l):
        result.add(indent & "  " & gl)
    else:
      result.add(indent & "  " & l)

proc genMicroKernelBlock(tokens: seq[Token], idx: var int, indent: string,
    blockIndent: int, localVars: var seq[string]): seq[string] =
  ## Native BybyLang syntax compiles directly to the micro-VM instruction API.
  result = @[]
  let header = tokens[idx].text.strip().strip(chars = {':'})
  let parts = header.split("->", maxsplit = 1)
  if parts.len != 2:
    raise newException(ValueError,
      "micro kernel syntax: micro kernel NAME INPUT -> OUTPUT threads N tg N out N:")
  let left = parts[0].splitWhitespace()
  let right = parts[1].splitWhitespace()
  if left.len < 2 or right.len < 1:
    raise newException(ValueError,
      "micro kernel needs a name, input array, and output variable")
  let kernelId = left[0]
  let inputName = left[1]
  let outputName = right[0]
  let threads = findNamedParam(right, "threads", "0")
  let tgSize = findNamedParam(right, "tg", "0")
  let outputElems = findNamedParam(right, "out", "0")
  if not kernelId[0].isAlphaAscii and kernelId[0] != '_':
    raise newException(ValueError, "invalid micro kernel name: " & kernelId)
  for c in kernelId:
    if not c.isAlphaNumeric and c != '_':
      raise newException(ValueError, "invalid micro kernel name: " & kernelId)

  idx.inc
  var body: seq[string] = @[]
  if idx < tokens.len and tokens[idx].indent > blockIndent:
    let bodyIndent = tokens[idx].indent
    while idx < tokens.len and tokens[idx].indent >= bodyIndent:
      let t = tokens[idx]
      if t.sym notin ["empty", "comment"]:
        body.add(t.text.strip())
      idx.inc
  if body.len == 0:
    raise newException(ValueError, "micro kernel '" & kernelId & "' has an empty body")

  if outputName notin localVars:
    result.add(indent & "var " & outputName & ": seq[float32] = @[]")
    localVars.add(outputName)
  let programName = "microProgram_" & kernelId
  result.add(indent & "block:")
  result.add(indent & "  var " & programName & " = VmProgram()")
  for instruction in body:
    let words = instruction.splitWhitespace()
    if words.len == 0 or not words[0].startsWith("emit"):
      raise newException(ValueError,
        "micro kernel body accepts VM emit instructions, got: " & instruction)
    let operands = if words.len > 1: words[1..^1].join(", ") else: ""
    let call = programName & "." & words[0] & "("
    result.add(indent & "  " & call & operands & ")")
  result.add(indent & "  " & outputName & " = executeMicroKernel(gpuBackendSelected, \"bybylang_" &
    kernelId & "\", " & programName & ", " & inputName & ", " & outputElems &
    ", " & threads & ", " & tgSize & ")")

proc genBlock(tokens: seq[Token], idx: var int, indent: string,
              funcNames: seq[string], localVars: var seq[string]): seq[string] =
  result = @[]
  let first = nextCode(tokens, idx)
  if first >= tokens.len:
    idx = first
    return

  let blockIndent = tokens[first].indent
  var tagMark = 0
  var tagTok: Token

  while idx < tokens.len and (tokens[idx].isTrivia or tokens[idx].indent >= blockIndent):
    let tk = tokens[idx]
    # Lines produced by the previous token belong to that token.
    tagLines(result, tagMark, tagTok)
    tagMark = result.len
    tagTok = tk

    if tk.indent > blockIndent:
      idx.inc
      continue

    case tk.sym
    of "empty":
      idx.inc

    of "comment":
      idx.inc

    of "import":
      idx.inc

    of "nimimport":
      # ALREADY collected and emitted once at the top of the file in generateNimCode -
      # skipped here to avoid emitting it twice or in the wrong place (Nim
      # does NOT allow "import" inside a proc body, only at the top-level
      # module scope).
      idx.inc

    of "print":
      traceDebug("Generating print: " & tk.text)
      result.add(indent & "echo " & tk.text["print".len..^1].strip())
      idx.inc

    of "microkernel":
      traceInfo("Generating native BybyLang micro kernel")
      result.add(genMicroKernelBlock(tokens, idx, indent, blockIndent, localVars))

    of "function":
      # Top-level functions are emitted by generateNimCode; a nested one is
      # emitted here as a local proc.
      let header = tk.text
      var fnVars = functionParams(header) & @["result"]
      result.add(indent & "proc " & functionSignature(header) & " =")
      idx.inc
      let body = nextCode(tokens, idx)
      if body < tokens.len and tokens[body].indent > tk.indent:
        result.add(genBlock(tokens, idx, indent & "  ", funcNames, fnVars))
      else:
        result.add(indent & "  discard")

    else:
      let line = tk.text.strip()

      if line.len == 0:
        idx.inc

      elif line.startsWith("#"):
        idx.inc

      elif line.startsWith("if ") or line.startsWith("elif ") or line == "else:":
        traceDebug("Generating conditional: " & line)
        result.add(genHeaderedBlock(tokens, idx, indent, line, blockIndent, funcNames, localVars))

      elif line.startsWith("while "):
        traceDebug("Generating while: " & line)
        result.add(genHeaderedBlock(tokens, idx, indent, line, blockIndent, funcNames, localVars))

      elif line.startsWith("for "):
        var forLine = line
        if "range(" in forLine:
          let inside = forLine.split("range(")[1].split(")")[0]
          let parts = inside.split(",")
          if parts.len == 2:
            forLine = forLine.replace("range(" & inside & ")",
                                       parts[0].strip() & ".." & parts[1].strip())
        traceDebug("Generating for: " & forLine)
        result.add(genHeaderedBlock(tokens, idx, indent, forLine, blockIndent, funcNames, localVars))

      elif line == "session:":
        traceDebug("Generating session block")
        result.add(genSessionBlock(tokens, idx, indent, blockIndent, localVars))

      elif line.startsWith("gpu "):
        for l in genGpuLine(line):
          result.add(indent & l)
        idx.inc

      elif line.startsWith("call "):
        let parts = line.split()
        if parts.len >= 2:
          let fname = parts[1]
          if fname in funcNames:
            result.add(indent & fname & "()")
          else:
            traceWarn("Function not found: " & fname)
            result.add(indent & "# [WARN] function not found: " & fname)
        idx.inc

      elif line in funcNames:
        result.add(indent & line & "()")
        idx.inc

      elif line.startsWith("mode is"):
        let parts = line.split()
        if parts.len >= 3:
          let m = parseIntSafe(parts[2])
          case m
          of 1: result.add(indent & "echo \"Mode 1: Low-level\"")
          of 2: result.add(indent & "echo \"Mode 2: Mid-level\"")
          of 3: result.add(indent & "echo \"Mode 3: High-level\"")
          of 4: result.add(indent & "echo \"Mode 4: Web-level\"")
          else: result.add(indent & "echo \"Unknown mode: " & $m & "\"")
        idx.inc

      elif line.startsWith("apu tran"):
        let parts = line.split("with")
        let name = stripQuotes(parts[0].split()[2].strip())
        let payload = parts[1].strip()
        result.add(indent & "apuTran(\"" & name & "\", " & payload & ")")
        idx.inc

      elif line.startsWith("apu mem"):
        let parts = line.split("with")
        let left = parts[0].split()
        let action = left[2]
        let target = stripQuotes(left[3])
        let value = parts[1].strip()
        result.add(indent & "apuMem(\"" & action & "\", \"" & target & "\", \"" & value & "\")")
        idx.inc

      elif line.startsWith("apu core"):
        result.add(indent & "apuCore(1, \"run\")")
        idx.inc

      elif line.startsWith("apu pin"):
        let words = line.split()
        result.add(indent & "apuPin(" & words[2] & ", \"" & words[4] & "\")")
        idx.inc

      elif line.startsWith("bit send"):
        result.add(indent & "bitSend(\"" & line.split()[2] & "\")")
        idx.inc

      elif line.startsWith("bit recv"):
        result.add(indent & "bitRecv()")
        idx.inc

      elif line.startsWith("mem map"):
        result.add(indent & "memMap(\"" & stripQuotes(line.split()[2]) & "\")")
        idx.inc

      elif line.startsWith("mem push"):
        let parts = line.split("with")
        result.add(indent & "memPush(\"" & stripQuotes(parts[0].split()[2]) & "\", \"" & parts[1].strip() & "\")")
        idx.inc

      elif line.startsWith("tran pulse"):
        let words = line.split()
        result.add(indent & "tranPulse(" & words[3] & ", \"" & words[^1] & "\")")
        idx.inc

      elif diagStatement(line) != "":
        # warn "text" / error "text" / traceback ["note"]: pass the .bybylang position along.
        let word = diagStatement(line)
        let rest = line[word.len..^1].strip()
        let where = ", " & dg.escapeLiteral(tk.file) & ", " & $tk.line & ")"
        case word
        of "warn": result.add(indent & "diagWarn(" & rest & where)
        of "error": result.add(indent & "diagError(" & rest & where)
        else: result.add(indent & "diagTraceback(" & (if rest.len > 0: rest else: "\"\"") & where)
        idx.inc

      elif isTypeSection(line):
        result.add(genRawBlock(tokens, idx, indent, blockIndent))

      elif isBlockHeader(line):
        result.add(genHeaderedBlock(tokens, idx, indent, line, blockIndent, funcNames, localVars))

      elif isDeclaration(line):
        for name in declaredNames(line):
          if name notin localVars: localVars.add(name)
        result.add(indent & line)
        idx.inc

      elif assignTarget(line).len > 0:
        # First plain "name = expr" declares the variable.
        let left = assignTarget(line)
        let right = line[line.find('=') + 1..^1].strip()
        if left notin localVars:
          localVars.add(left)
          result.add(indent & "var " & left & " = " & right)
        else:
          result.add(indent & left & " = " & right)
        idx.inc

      else:
        result.add(indent & line)
        idx.inc
  tagLines(result, tagMark, tagTok)
# --------------------------
# Translate Nim's diagnostics on the generated file back to .bybylang positions
# --------------------------
var rawNimOutput = false        ## --raw-nim: also print Nim's own output
var showNimWarnings = false     ## --nim-warnings: report Nim warnings of the generated code

type NimDiag = object
  path, sev, msg: string
  line, col: int
  notes: seq[string]

proc parseNimOutput(output: string): seq[NimDiag] =
  for raw in output.splitLines:
    let ln = raw.strip(trailing = true)
    var matched = false
    let lp = ln.find('(')
    let rp = if lp > 0: ln.find(')', lp) else: -1
    if lp > 0 and rp > lp and rp + 2 < ln.len and ln[rp + 1] == ' ':
      let inside = ln[lp + 1 ..< rp].split(", ")
      let colon = ln.find(':', rp + 2)
      if inside.len == 2 and colon > rp:
        try:
          result.add NimDiag(path: ln[0 ..< lp], line: parseInt(inside[0]), col: parseInt(inside[1]),
                             sev: ln[rp + 2 ..< colon], msg: ln[colon + 1 .. ^1].strip())
          matched = true
        except ValueError: discard
    if not matched and result.len > 0 and ln.strip().len > 0 and
        not ln.startsWith("Hint:") and not ln.allCharsInSet({'.', ' '}):
      result[^1].notes.add ln.strip()

var gSrcCache = initTable[string, seq[string]]()

proc sourceLines(file: string): seq[string] =
  if file notin gSrcCache:
    gSrcCache[file] = if fileExists(file): readFile(file).splitLines else: @[]
  gSrcCache[file]

proc markerOf(nimLine: string): tuple[line, file: int] =
  ## Parse the trailing `#bb:<line>:<file>` marker; (0, -1) when there is none.
  let at = nimLine.rfind(" #bb:")
  if at < 0: return (0, -1)
  let parts = nimLine[at + 5 .. ^1].split(':')
  try: (parseInt(parts[0]), parseInt(parts[1])) except ValueError: (0, -1)

proc stripMarker(nimLine: string): string =
  let at = nimLine.rfind(" #bb:")
  if at < 0: nimLine else: nimLine[0 ..< at]

proc reportNimOutput(output, nimFile: string): int =
  ## Print the diagnostics that belong to the generated file as clang-style messages on the
  ## .bybylang source. Returns the number of errors shown.
  let nimLines = readFile(nimFile).splitLines
  for d in parseNimOutput(output):
    let ours = d.path.extractFilename == nimFile.extractFilename
    let isError = d.sev == "Error"
    if not isError and not (ours and showNimWarnings and d.sev == "Warning"): continue
    if not ours:
      # an error inside a library module: no .bybylang position exists
      dg.diagEmit(dg.dsError, d.path, d.line, d.col, d.msg, notes = d.notes)
      inc result
      continue
    var at = d.line
    var mk = if at >= 1 and at <= nimLines.len: markerOf(nimLines[at - 1]) else: (0, -1)
    var useEnd = false
    # "expected ')'" and indentation errors are found at the token AFTER the mistake: blame the
    # end of the previous statement, like a C compiler does.
    if at >= 1 and at <= nimLines.len and
        (d.msg.startsWith("expected") or d.msg.startsWith("invalid indentation")):
      let gen = stripMarker(nimLines[at - 1])
      let genIndent = gen.len - gen.strip(trailing = false).len
      var got = ""
      let gotAt = d.msg.find("but got: '")
      if gotAt >= 0: got = d.msg[gotAt + 10 .. ^1].strip(chars = {'\''})
      if (got.len > 0 and gen.strip().startsWith(got)) or d.col <= genIndent + 1:
        var j = at - 1
        while j >= 1 and markerOf(nimLines[j - 1]).file < 0: dec j
        if j >= 1:
          mk = markerOf(nimLines[j - 1])
          useEnd = true
    if mk.file < 0:
      var j = at
      while j >= 1 and markerOf(nimLines[j - 1]).file < 0: dec j
      if j >= 1: mk = markerOf(nimLines[j - 1])
    if mk.file < 0 or mk.file >= gSrcFiles.len:
      dg.diagEmit(dg.dsError, "(generated)", d.line, d.col, d.msg, notes = d.notes)
      inc result
      continue
    let file = gSrcFiles[mk.file]
    let src = sourceLines(file)
    let srcText = if mk.line >= 1 and mk.line <= src.len: src[mk.line - 1] else: ""
    let srcIndent = srcText.len - srcText.strip(trailing = false).len
    var col = srcIndent + 1
    var width = max(1, srcText.strip().len)
    if useEnd:
      col = srcText.strip(leading = false).len + 1
      width = 1
    elif at >= 1 and at <= nimLines.len:
      let gen = stripMarker(nimLines[at - 1])
      let genIndent = gen.len - gen.strip(trailing = false).len
      if gen.strip() == srcText.strip():
        col = min(max(srcIndent + (d.col - genIndent), srcIndent + 1), srcText.len)
        width = 1
    dg.diagEmit(dg.dsError, file, mk.line, col, d.msg, srcText, width, d.notes)
    inc result

# --------------------------
# Generate Nim code + compile to binary release
# --------------------------
proc generateNimCode(tokens: seq[Token], outFile: string) =
  traceInfo("Generating Nim code for: " & outFile)
  # Split the program into top-level type/const declarations, functions (in
  # source order, so a function may call any function defined above it) and
  # the remaining top-level statements. Declarations come first in the
  # output so functions can use the types.
  var funcHeaders = initOrderedTable[string, string]()
  var funcBodiesLocal = initOrderedTable[string, seq[Token]]()
  var funcTokens = initTable[string, Token]()
  var declTokens: seq[Token] = @[]
  var mainTokens: seq[Token] = @[]
  var idx = 0

  proc takeBlock(tokens: seq[Token], idx: var int, baseIndent: int): seq[Token] =
    while idx < tokens.len:
      let j = nextCode(tokens, idx)
      if j >= tokens.len or tokens[j].indent <= baseIndent: break
      while idx <= j:
        result.add(tokens[idx])
        idx.inc

  while idx < tokens.len:
    let t = tokens[idx]
    if t.sym == "function" and t.indent == 0:
      let fname = functionName(t.text)
      idx.inc
      let body = takeBlock(tokens, idx, t.indent)
      if fname in funcBodiesLocal:
        traceWarn("Function redefined: " & fname)
      funcHeaders[fname] = t.text
      funcTokens[fname] = t
      funcBodiesLocal[fname] = body
      traceDebug("Found function: " & fname & " with " & $body.len & " body tokens")
    elif t.indent == 0 and t.sym == "other" and
        (t.text.isTypeSection or t.text.startsWith("const ")):
      declTokens.add(t)
      idx.inc
      declTokens.add(takeBlock(tokens, idx, t.indent))
    else:
      mainTokens.add(t)
      idx.inc

  # --- create the file ---
  var nimFile = outFile
  if not nimFile.endsWith(".nim"): nimFile &= ".nim"
  traceInfo("Writing to: " & nimFile)

  var code = newSeq[string]()
  code.add("import strutils, sequtils")
  code.add("import bybylang")
  code.add("import diag")
  # NEW: collects ALL "import nim \"X\"" from every token (including files
  # imported through resolveImports - the nimimport tokens survive
  # resolution) - emitted ONCE at the top
  # of the file, no duplicates, BEFORE any proc/top-level code (Nim requires
  # imports at module scope, the earlier the safer).
  var nimImportsSeen = initTable[string, bool]()
  for t in tokens:
    if t.sym == "nimimport" and t.text.len > 0 and t.text notin nimImportsSeen:
      nimImportsSeen[t.text] = true
      code.add("import " & t.text)
  code.add("const RAM_SIZE = 1024")
  code.add("var RAM: array[0..RAM_SIZE-1, int]")
  code.add("var BUS: seq[string] = @[]")
  code.add("var Pins: array[0..31, bool]")
  code.add("")
  code.add("proc stripQuotes(s: string): string =")
  code.add("  if s.len >= 2 and s[0] == '\"' and s[^1] == '\"':")
  code.add("    return s[1..^2]")
  code.add("  else:")
  code.add("    return s")
  code.add("")
  # --- proc HW ---
  code.add("proc apuTran(name: string, payload: string) =")
  code.add("  BUS.add(payload)")
  code.add("  echo \"[APU-TRAN] \", name, \" -> \", payload")
  code.add("")
  code.add("proc apuMem(action: string, target: string, value: string) =")
  code.add("  var ramAddr = parseInt(target.replace(\"RAM\", \"\"))")
  code.add("  if action == \"write\":")
  code.add("    RAM[ramAddr] = parseInt(value)")
  code.add("  elif action == \"read\":")
  code.add("    echo \"[APU-MEM] RAM[\", ramAddr, \"] -> \", RAM[ramAddr]")
  code.add("")
  code.add("proc apuCore(mode: int, code: string) =")
  code.add("  echo \"[APU-CORE] Mode:\", mode, \" run:\", code")
  code.add("")
  code.add("proc apuPin(pin: int, state: string) =")
  code.add("  Pins[pin] = (state == \"high\")")
  code.add("")
  code.add("proc bitSend(bits: string) =")
  code.add("  BUS.add(bits)")
  code.add("")
  code.add("proc bitRecv() =")
  code.add("  if BUS.len > 0:")
  code.add("    echo BUS[0]")
  code.add("    delete(BUS, 0)")
  code.add("  else:")
  code.add("    echo \"[BIT-RECV] empty\"")
  code.add("")
  code.add("proc memMap(target: string) =")
  code.add("  echo \"[MEM-MAP] \", target")
  code.add("")
  code.add("proc memPush(target: string, value: string) =")
  code.add("  echo \"[MEM-PUSH] \", target, \" <- \", value")
  code.add("")
  code.add("proc tranPulse(pin: int, width: string) =")
  code.add("  echo \"[TRAN-PULSE] pin \", pin, \" width \", width")
  code.add("")

  # --- collect the function names ---
  var funcNames: seq[string] = @[]
  for k, _ in funcBodiesLocal:
    funcNames.add(k)

  # --- 1. Top-level type/const declarations ---
  if declTokens.len > 0:
    var declVars: seq[string] = @[]
    var didx = 0
    code.add("")
    for l in genBlock(declTokens, didx, "", funcNames, declVars): code.add(l)

  # --- 2. Functions, in source order ---
  traceInfo("Generating " & $funcBodiesLocal.len & " functions")
  for k, v in funcBodiesLocal:
    code.add("")
    var header = @["proc " & functionSignature(funcHeaders[k]) & " ="]
    tagLines(header, 0, funcTokens[k])
    code.add(header[0])
    var localVars = functionParams(funcHeaders[k]) & @["result"]
    var bidx = 0
    let bodyLines = genBlock(v, bidx, "  ", funcNames, localVars)
    if bodyLines.len == 0:
      code.add("  discard")
    else:
      for l in bodyLines: code.add(l)

  # --- 3. Top-level statements ---
  code.add("")
  var topVars: seq[string] = @[]
  var tidx = 0
  if mainTokens.len > 0:
    traceInfo("Generating top-level code")
    let topLines = genBlock(mainTokens, tidx, "", funcNames, topVars)
    for l in topLines: code.add(l)

  writeFile(nimFile, code.join("\n"))
  traceInfo("Generated " & $code.len & " lines of Nim code")
  
  let nimcacheDir = getEnv("BYBYLANG_NIMCACHE").strip()
  let nimcacheArg = if nimcacheDir.len > 0:
      " --nimcache:" & quoteShell(nimcacheDir)
    else: ""
  # `traceback` needs Nim stack frames, which -d:release drops; turn them on only when it is used.
  var usesTraceback = false
  for t in tokens:
    if t.sym == "other" and (t.text == "traceback" or t.text.startsWith("traceback ")):
      usesTraceback = true
  let stackFlags = if usesTraceback: " --stackTrace:on --lineTrace:on" else: ""
  # BYBYLANG_NIM_FLAGS adds Nim options (for example a cross-compile: --cpu:arm64 --passC:"-arch arm64").
  let extraFlags = getEnv("BYBYLANG_NIM_FLAGS").strip()
  let cmd = "nim c -d:release --hints:off --colors:off" & stackFlags &
    (if extraFlags.len > 0: " " & extraFlags else: "") & " --path:" & quoteShell(getAppDir()) &
    nimcacheArg & " -o:" & quoteShell(outFile) & " " & quoteShell(nimFile)
  traceInfo("Compiling with: " & cmd)
  let res = execCmdEx(cmd)
  if rawNimOutput: echo res.output
  let shown = reportNimOutput(res.output, nimFile)
  if res.exitCode != 0:
    if shown == 0 and not rawNimOutput:
      # Nim failed without a diagnostic we can place: show what it said.
      dg.diagEmit(dg.dsError, "", 0, 0, "Nim failed to build the generated code (exit " &
        $res.exitCode & ")", notes = @[res.output.strip()])
    stderr.writeLine $max(shown, 1) & " error(s) generated."
    quit(1)
  traceInfo("Built executable: " & outFile)

# --------------------------
# Main
# --------------------------
proc main() =
  var inputFile = ""
  var args = commandLineParams()
  var aotFile = ""
  
  for i, a in args:
    if a == "--ignore-errors":
      ignoreErrors = true
    elif a == "--quiet":
      quietMode = true
    elif a == "-v" or a == "--verbose":
      gTraceLevel = tlInfo
    elif a == "-vv":
      gTraceLevel = tlDebug
    elif a == "--raw-nim":
      rawNimOutput = true
    elif a == "--nim-warnings":
      showNimWarnings = true
    elif a == "-Werror":
      dg.gDiagWarningsAsErrors = true
    elif a.startsWith("--aot="):
      aotFile = a.split('=')[1]
    elif a == "--help" or a == "-h":
      echo "Usage: ./bybylang <file.bybylang> [options]"
      echo ""
      echo "Options:"
      echo "  --aot=FILE          Compile to executable FILE"
      echo "  --ignore-errors     Ignore non-fatal errors"
      echo "  --quiet             Suppress informational output"
      echo "  -v, -vv             Verbose compiler log (info / debug)"
      echo "  --raw-nim           Also print the Nim compiler's own output"
      echo "  --nim-warnings      Report Nim warnings of the generated code"
      echo "  -Werror             Treat warnings as errors"
      echo "  --help, -h          Show this help"
      quit(0)
    elif i == 0 or inputFile == "":
      inputFile = a

  if inputFile == "":
    echo "Usage: ./bybylang <file.bybylang> [--ignore-errors] [--quiet] [--aot=output]"
    quit(1)

  if not fileExists(inputFile):
    echo "[ERROR] File not found: ", inputFile
    quit(1)

  traceInfo("Starting BybyLang with file: " & inputFile)
  traceDebug("Arguments: " & $args)

  let tokens = tokenizeFile(inputFile)
  if dg.gDiagErrors > 0 and not ignoreErrors:
    stderr.writeLine $dg.gDiagErrors & " error(s) generated."
    quit(1)
  traceInfo("Tokenization complete, " & $tokens.len & " tokens")

  if aotFile != "":
    traceInfo("Generating AOT executable: " & aotFile)
    generateNimCode(tokens, aotFile)
  else:
    traceInfo("No AOT output specified")

  traceInfo("BybyLang finished")

when isMainModule:
  try:
    main()
  except CatchableError as e:
    # A compiler failure is reported as a diagnostic; the Nim stack trace only with -v.
    dg.diagEmit(dg.dsError, "", 0, 0, e.msg)
    if gTraceLevel <= tlInfo: stderr.writeLine e.getStackTrace()
    quit(1)
