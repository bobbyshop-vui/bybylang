## Dispatch recorder (`fprobe 2`), with optional exact upload data sidecar.
##
## One `op` line is written per kernel launch: a fixed shape
##   op oN tag=NAME in=(port=bX@byte:bytes,...) out=(...) attrs(...)
##      impl=session:PROC | impl=micro:HASH | impl=ext:OP parent=... prog=...
## Text records contain structure, sampled fingerprints and short tables. With
## BYBY_FIR_PROBE_DATA=1, host upload bytes stream to a separate binary file so
## unnamed auxiliary buffers can be replayed without hooks in model code.
##
## Enabled with BYBY_FIR_PROBE=<path>; step selection with
## BYBY_FIR_PROBE_STEPS=decode:N,prefill:M. When disabled every hook is one
## predictable boolean test.
import std/[os, strutils, tables, json, macros, exitprocs, locks]
import ./probe_data

const
  probeBuild = "nim-" & NimVersion
  probeVersion = "fprobe 2"

const probeSmallLimit* = 64

var firProbeOn* = false

type
  BufferRecord = object
    bytes: int
    origin: string
    names: seq[string]
  OpDraft = object
    api, tag, ins, attrs, impl, prog: string
    parent: int
  EncoderBinding = object
    raw: pointer
    offset, bytes: int
  ProbeStateObj = object
    path, backendName: string
    events: seq[string]
    buffers: seq[BufferRecord]
    identities: Table[uint64, int]
    rawIdentities: Table[pointer, uint64]
    nextIdentity: uint64
    programs: seq[seq[uint32]]
    programIds: Table[seq[uint32], int]
    pipelineWords: Table[string, seq[uint32]]
    codeWords: Table[uint64, seq[uint32]]
    vmArenas: Table[uint64, bool]
    driverUploads: Table[uint64, bool]
    drafts: Table[int, OpDraft]
    encoderPipeline: string
    encoderBindings: Table[int, EncoderBinding]
    encoderTgBytes: int
    nextOp, nextDisp, nextTransfer, nextAutoFlush: int
    captureFile: File
    captureOpen: bool
    seenStep, selectedStep, insideStep: bool
    decodeLimit, prefillLimit, decodeSeen, prefillSeen: int
  ProbeState = ref ProbeStateObj

# ARC/ORC destroy managed module globals before C atexit callbacks run. Pin
# the entire state until our exit callback, and keep only an unmanaged pointer
# in the module globals. Ordinary string/seq/table ownership then remains valid.
var state: ptr ProbeStateObj
var recorderLock: Lock
var recorderLockReady = false
var lockDepth {.threadvar.}: int
var activeOp {.threadvar.}: int  # op id + 1; zero means no parent on a new thread.
var cpuArena {.threadvar.}: string

template currentOp: int = activeOp - 1

template probeLocked() =
  if lockDepth == 0: acquire(recorderLock)
  inc lockDepth
  defer:
    dec lockDepth
    if lockDepth == 0: release(recorderLock)

proc probeQuote*(s: string): string = $(%s)
proc probeFloat*(v: float32): string = "h" & toHex(cast[uint32](v), 8).toLowerAscii
proc probeFloat*(v: float64): string = "h" & toHex(cast[uint64](v), 16).toLowerAscii
proc probeShape*(shape: openArray[int]): string =
  result = "("
  for i, n in shape:
    if i > 0: result.add ' '
    result.add $n
  result.add ')'

proc probeFlush*() {.gcsafe.}
var probeExitRegistered = false

proc capturing(): bool = not state.seenStep or (state.insideStep and state.selectedStep)
proc probeEvent*(line: string) =
  probeLocked()
  if capturing():
    state.events.add line
    # Keep a usable trace on disk if the process is terminated during a recording.
    # The first event creates the file; later rewrites grow geometrically, so total
    # output written stays amortised linear in the number of events.
    if state.events.len >= state.nextAutoFlush or
        (state.nextDisp == 0 and line.startsWith("disp ")):
      state.nextAutoFlush = state.events.len + state.events.len div 4 + 64
      probeFlush()

proc probeBackend*(name: string) =
  probeLocked()
  if name notin ["auto", "unknown"]: state.backendName = name

proc probeNewIdentity*(): uint64 =
  probeLocked()
  inc state.nextIdentity
  state.nextIdentity

proc probeIdentity*(raw: pointer): uint64 =
  probeLocked()
  if raw == nil: return probeNewIdentity()
  if not state.rawIdentities.hasKey(raw): state.rawIdentities[raw] = probeNewIdentity()
  state.rawIdentities[raw]

proc probeAlias*(raw: pointer; identity: uint64) =
  probeLocked()
  if raw != nil: state.rawIdentities[raw] = identity

proc probeForget*(raw: pointer) =
  probeLocked()
  state.rawIdentities.del(raw)

proc probeBuffer*(identity: uint64; bytes: int; origin = "observed"): string =
  probeLocked()
  var id: int
  if not state.identities.hasKey(identity):
    id = state.buffers.len
    state.identities[identity] = id
    state.buffers.add BufferRecord(bytes: bytes, origin: origin)
  else:
    id = state.identities[identity]
    if bytes > state.buffers[id].bytes: state.buffers[id].bytes = bytes
  "b" & $id

proc probeResize*(id: string; bytes: int) =
  probeLocked()
  ## Late size knowledge (GPU handles learn their extent at alloc/upload).
  if id.len < 2 or id[0] != 'b': return
  let index = parseInt(id[1 .. ^1])
  if index < 0 or index >= state.buffers.len: return
  if bytes > state.buffers[index].bytes: state.buffers[index].bytes = bytes

proc probeNamed*(id: string; name, dtype: string; shape: openArray[int]; offset, bytes: int) =
  probeLocked()
  # Multiple weights can occupy disjoint slices of the same arena.
  let index = parseInt(id[1 .. ^1])
  if offset == 0 and bytes == state.buffers[index].bytes:
    state.buffers[index].names.add " name=" & probeQuote(name) & " dtype=" & dtype &
      " shape=" & probeShape(shape)
  probeEvent("name " & id & "@" & $offset & ":" & $bytes & " name=" & probeQuote(name) &
    " dtype=" & dtype & " shape=" & probeShape(shape))

proc probeRoleRecord*(id: string; role: string; offset, bytes: int) =
  probeLocked()
  probeEvent("role " & id & "@" & $offset & ":" & $bytes & " role=" & probeQuote(role))

proc probeScalar*(id: string; role: string; offset, value: int) =
  probeLocked()
  if role notin ["token", "position"]:
    raise newException(ValueError, "probe scalar role must be token or position")
  probeEvent("upload u" & $state.nextTransfer & " dst=" & id & "@" & $offset & ":4 role=" &
    probeQuote(role) & " value=" & $value & " op=o" & $currentOp)
  inc state.nextTransfer

proc probeSmallData*[T](data: openArray[T]): string =
  ## ` data=` + hex bit patterns of a short 4-byte table, else nothing.
  when sizeof(T) == 4:
    if data.len == 0 or data.len > probeSmallLimit: return ""
    var parts: seq[string]
    for v in data:
      var bits: uint32
      copyMem(addr bits, unsafeAddr v, 4)
      parts.add toHex(BiggestInt(bits), 8).toLowerAscii()
    " data=" & parts.join(",")
  else:
    ""

proc probeCapture*(data: pointer; bytes: int): string =
  ## Fingerprints are cheap; optional exact bytes also preserve auxiliary uploads
  ## (composite arenas, LUTs, adapters) that need not exist in a checkpoint.
  if not firProbeOn or data == nil or bytes <= 0: return ""
  probeLocked()
  if not capturing(): return ""
  result = " fingerprint=" & fingerprint(data, bytes)
  if getEnv("BYBY_FIR_PROBE_DATA", "0") != "1": return
  # An optional positive cap omits large payloads; fingerprints remain available
  # for checkpoint matching. Zero (the default) keeps the full auxiliary data.
  let limit = try: parseInt(getEnv("BYBY_FIR_PROBE_DATA_MAX", "0"))
              except ValueError: -1
  if limit > 0 and bytes > limit: return
  if not state.captureOpen:
    state.captureFile = open(state.path & ".data", fmWrite)
    state.captureOpen = true
  let offset = state.captureFile.getFilePos()
  if state.captureFile.writeBuffer(data, bytes) != bytes:
    raise newException(IOError, "cannot write FIR probe upload data")
  state.captureFile.flushFile()
  result.add " capture=" & $offset & ":" & $bytes

proc probeCapture*[T](data: openArray[T]): string =
  if data.len > 0: probeCapture(unsafeAddr data[0], data.len*sizeof(T)) else: ""

proc probeWordHashValue(words: openArray[uint32]): uint64 =
  result = 0xcbf29ce484222325'u64
  for word in words:
    for i in 0 .. 3:
      result = (result xor uint64((word shr (i*8)) and 0xff)) * 0x100000001b3'u64

proc probeWordHash*(words: openArray[uint32]): string =
  toHex(probeWordHashValue(words), 16).toLowerAscii

proc probeProgram*(words: openArray[uint32]): string =
  probeLocked()
  # Keep an immutable owned copy: callers may reuse or mutate their bytecode.
  let owned = @words
  if not state.programIds.hasKey(owned):
    state.programIds[owned] = state.programs.len
    state.programs.add owned
  "p" & $state.programIds[owned]

proc probeCode*(identity: uint64; words: openArray[uint32]) =
  probeLocked()
  state.codeWords[identity] = @words

proc probeCodeProgram*(identity: uint64): string =
  probeLocked()
  if state.codeWords.hasKey(identity): return probeProgram(state.codeWords[identity])
  "none"

proc probePipeline*(name: string; words: openArray[uint32]) =
  probeLocked()
  # Called even on a warm JIT cache; this table belongs to the recorder.
  state.pipelineWords[name] = @words

proc probePipelineProgram*(name: string): string =
  probeLocked()
  if state.pipelineWords.hasKey(name): return probeProgram(state.pipelineWords[name])
  probeEvent("note kind=missing_words pipeline=" & probeQuote(name))
  "none"

proc probeProgramImpl*(prog: string): string =
  probeLocked()
  if prog.len < 2 or prog[0] != 'p': return "micro:none"
  let index = parseInt(prog[1 .. ^1])
  if index < 0 or index >= state.programs.len: return "micro:none"
  "micro:" & probeWordHash(state.programs[index])

proc probeProgramWords*(prog: string): seq[uint32] =
  probeLocked()
  if prog.len < 2 or prog[0] != 'p': return @[]
  let index = parseInt(prog[1 .. ^1])
  if index < 0 or index >= state.programs.len: return @[]
  result = @(state.programs[index].toOpenArray(0, state.programs[index].high))

proc probeTagFor*(api: string; opName = ""): string =
  ## ATen identity of a semantic session call. Unknown calls keep a `fir::`
  ## name; the lifter reports them instead of dropping them.
  case api
  of "sessionLayernormEnc": "aten::native_layer_norm"
  of "sessionRmsnormEnc": "aten::rms_norm"
  of "sessionMatmul", "sessionMatmulNT", "sessionMatmulQ4",
     "sessionMatmulNTAccelerated", "sessionMatmulNTFp16Coop",
     "sessionFp4GemmTensorCore", "sessionFp4GemmTiledEnc",
     "sessionMatmulNTTiledEnc": "aten::matmul"
  of "sessionSoftmax": "aten::softmax"
  of "sessionAttentionFusedEnc", "sessionAttentionDecodeGqaEnc",
     "sessionAttentionDecodeAllHeadsEnc", "sessionFlashAttentionPrefillEnc":
    "aten::scaled_dot_product_attention"
  of "sessionKvScatterEnc": "fir::kv_scatter"
  of "sessionRope": "fir::rotary_embedding"
  of "sessionEmbeddingLookupEnc": "aten::embedding"
  of "sessionGatherEnc": "aten::gather"
  of "sessionCompareEnc": "aten::lt"
  of "sessionArgmaxIndexEnc": "aten::argmax"
  of "sessionPermuteEnc": "aten::permute"
  of "sessionSplitSliceEnc": "aten::slice"
  of "sessionScale": "aten::mul"
  of "sessionCopyInto": "aten::copy_"
  of "sessionActivation", "sessionVecOp":
    if opName.len > 0: "aten::" & opName else: "aten::elementwise"
  of "sessionUpload", "sessionUploadBorrowed", "sessionUploadPrivate",
     "sessionUploadIndices", "sessionUploadPrivatePtr", "sessionWrapHostRange":
    "fir::upload"
  of "sessionRead": "fir::read"
  of "sessionMamba2SsdPrefillEnc": "fir::ssd_prefill"
  of "sessionBegin", "sessionEnd", "sessionCommitContinue", "sessionFree",
     "sessionJitPrewarm", "sessionJitMicroVmEnc", "sessionMicroVmEnc":
    "fir::launch"
  else: "fir::" & api

proc probeOpStart*(api, tag, ins, attrs, impl, prog: string): tuple[previous, id: int] =
  probeLocked()
  result = (currentOp, state.nextOp)
  state.drafts[state.nextOp] = OpDraft(api: api, tag: tag, ins: ins, attrs: attrs, impl: impl,
    prog: prog, parent: currentOp)
  activeOp = state.nextOp + 1
  inc state.nextOp

proc probeOpFinish*(context: tuple[previous, id: int]; outs, extraAttrs, resultValue: string) =
  probeLocked()
  ## Closes the op once the output handles are final, so one `op` line carries
  ## the complete input/output set.
  let id = context.id
  var d: OpDraft
  if state.drafts.hasKey(id):
    d = state.drafts[id]
    state.drafts.del(id)
  else:
    d = OpDraft(api: "", tag: "fir::unknown", ins: "", attrs: "", impl: "session:unknown", prog: "none")
  var attrs = d.attrs
  if extraAttrs.len > 0:
    if attrs.len > 0: attrs.add ","
    attrs.add extraAttrs
  if resultValue.len > 0:
    if attrs.len > 0: attrs.add ","
    attrs.add "result=" & resultValue
  let parent = if d.parent < 0: "none" else: "o" & $d.parent
  if capturing():
    var inText = d.ins
    if inText.len > 0 and inText[0] == ',': inText = inText[1 .. ^1]
    var outText = outs
    if outText.len > 0 and outText[0] == ',': outText = outText[1 .. ^1]
    var attrText = attrs
    if attrText.len > 0 and attrText[0] == ',': attrText = attrText[1 .. ^1]
    probeEvent("op o" & $id & " tag=" & d.tag & " in=(" & inText & ") out=(" & outText &
      ") attrs(" & attrText & ") impl=" & d.impl & " parent=" & parent & " prog=" & d.prog)
  activeOp = context.previous + 1

proc probeGpuDispatch*(prog, pipe: string; nThreads, tgSize, tgBytes: int; bindings = "") =
  probeLocked()
  if not capturing(): return
  var program = prog
  if program.len == 0 or program == "none":
    program = probePipelineProgram(pipe)
  probeEvent("disp d" & $state.nextDisp & " prog=" & program & " nThreads=" & $nThreads &
    " tgSize=" & $tgSize & " tgbytes=" & $tgBytes & " pipe=" & probeQuote(pipe) &
    (if bindings.len > 0: " " & bindings else: "") &
    " op=" & (if currentOp < 0: "none" else: "o" & $currentOp))
  inc state.nextDisp

proc probeEncoderPipeline*(name, backend: string) =
  probeLocked()
  probeBackend(backend)
  state.encoderPipeline = name
  state.encoderBindings.clear()
  state.encoderTgBytes = 0

proc probeEncoderBind*(index: int; raw: pointer; offset = 0; bytes = -1) =
  probeLocked()
  state.encoderBindings[index] = EncoderBinding(raw: raw, offset: offset, bytes: bytes)

proc probeEncoderTgBytes*(bytes: int) =
  probeLocked()
  state.encoderTgBytes = bytes

proc probeEncoderDispatch*(nThreads, tgSize: int; tgBytes = -1) =
  probeLocked()
  var bindings = "bind=("
  for index, binding in state.encoderBindings:
    if bindings.len > 6: bindings.add ','
    let id = probeBuffer(probeIdentity(binding.raw), max(0, binding.bytes))
    let bytes = if binding.bytes >= 0: binding.bytes
                else: state.buffers[parseInt(id[1 .. ^1])].bytes
    bindings.add "b" & $index & ":" & id & "@" & $binding.offset & ":" &
      $max(0, bytes - binding.offset)
  bindings.add ')'
  probeGpuDispatch(probePipelineProgram(state.encoderPipeline), state.encoderPipeline,
    nThreads, tgSize, (if tgBytes >= 0: tgBytes else: state.encoderTgBytes), bindings)

proc probeDebugPath*(): string =
  if state == nil: return ""
  probeLocked()
  state.path

proc probeFlush*() =
  if state == nil: return
  probeLocked()
  if state.path.len == 0: return
  var output = probeVersion & "\nbackend " & state.backendName & " device=\"unspecified\" build=" &
    probeQuote(probeBuild) & " launch=\"session\"\n"
  for i, b in state.buffers:
    output.add "buf b" & $i & " bytes=" & $b.bytes & " origin="
    output.add b.origin
    for name in b.names: output.add name
    output.add '\n'
  for i, program in state.programs:
    output.add "prog p" & $i & " kind=vm words=" & $program.len &
      " hash=" & probeWordHash(program) & "\n"
  for event in state.events:
    output.add event
    output.add '\n'
  for i, program in state.programs:
    output.add "words p" & $i & "\n"
    for j, word in program:
      output.add " " & toHex(word, 8).toLowerAscii
      if j mod 16 == 15 or j == program.len - 1: output.add '\n'
  output.add "end\n"
  # Rename a complete snapshot in the same directory. SIGTERM may interrupt a
  # rewrite, but the published trace always remains a complete previous snapshot.
  let pending = state.path & ".tmp"
  writeFile(pending, output)
  moveFile(pending, state.path)

proc firProbeStop*() =
  ## Manual stop (also used when a new recording starts).
  if state == nil: return
  probeLocked()
  if firProbeOn:
    probeFlush()
    firProbeOn = false
  if state.captureOpen:
    state.captureFile.close()
    state.captureOpen = false

proc firProbeShutdown() {.noconv.} =
  probeLocked()
  try:
    if firProbeOn and state != nil and state.events.len > 0:
      probeFlush()
  finally:
    firProbeOn = false
    if state != nil:
      if state.captureOpen: state.captureFile.close()
      GC_unref(cast[ProbeState](state))
      state = nil

proc firProbeStart*(path: string; steps = "") =
  # No recorder allocation or synchronization is needed on the disabled path.
  if not recorderLockReady:
    initLock(recorderLock)
    recorderLockReady = true
  probeLocked()
  firProbeStop()
  if path.len == 0: return
  if path.contains('\n') or path.contains('\r') or path.contains('\0'):
    raise newException(ValueError, "BYBY_FIR_PROBE must be a single path line")
  let dir = parentDir(path)
  if dir.len > 0 and not dirExists(dir):
    raise newException(IOError, "BYBY_FIR_PROBE directory does not exist: " & dir)
  if state == nil:
    let owned = ProbeState()
    GC_ref(owned)
    state = cast[ptr ProbeStateObj](owned)
  state.events.setLen(0); state.buffers.setLen(0)
  state.identities.clear(); state.rawIdentities.clear()
  state.vmArenas.clear()
  state.driverUploads.clear()
  state.programs.setLen(0); state.programIds.clear(); state.drafts.clear()
  # Prepared JIT pipelines and code buffers may outlive a recording. Preserve
  # their bytecode registries across start/stop; neither contains tensor data.
  state.backendName = "unknown"
  state.encoderPipeline = ""; state.encoderBindings.clear(); state.encoderTgBytes = 0
  activeOp = 0; state.nextOp = 0; state.nextDisp = 0; state.nextTransfer = 0
  state.seenStep = false; state.selectedStep = false; state.insideStep = false
  state.decodeSeen = 0; state.prefillSeen = 0; state.decodeLimit = -1; state.prefillLimit = -1
  cpuArena = ""
  state.nextAutoFlush = 1
  if steps.len > 0:
    state.decodeLimit = 0; state.prefillLimit = 0
    for part in steps.split(','):
      let pair = part.split(':')
      if pair.len != 2 or pair[0] notin ["decode", "prefill"]:
        raise newException(ValueError, "invalid BYBY_FIR_PROBE_STEPS: " & steps)
      let n = parseInt(pair[1])
      if n < 0: raise newException(ValueError, "negative probe step limit")
      if pair[0] == "decode": state.decodeLimit = n else: state.prefillLimit = n
  state.path = path
  if not probeExitRegistered:
    addExitProc(firProbeShutdown)
    probeExitRegistered = true
  firProbeOn = true
  let od = getEnv("BYBY_ONE_DISPATCH", "0")
  if od != "0":
    probeEvent("note kind=one_dispatch value=" & probeQuote(od) & " detail=\"opaque fast path; disable for lifting\"")

proc probeStepBegin*(kind: string; index, position, token: int; tokens = 1) =
  if not firProbeOn: return
  probeLocked()
  if state.insideStep: raise newException(ValueError, "nested probe step")
  if kind notin ["prefill", "decode"]: raise newException(ValueError, "invalid probe step kind")
  state.seenStep = true; state.insideStep = true
  if kind == "decode":
    state.selectedStep = state.decodeLimit < 0 or state.decodeSeen < state.decodeLimit
    inc state.decodeSeen
  else:
    state.selectedStep = state.prefillLimit < 0 or state.prefillSeen < state.prefillLimit
    state.prefillSeen += tokens
  probeEvent("step " & kind & " index=" & $index & " position=" & $position &
    " token=" & $token & " tokens=" & $tokens)

proc probeStepEnd*() =
  if not firProbeOn: return
  probeLocked()
  if not state.insideStep: raise newException(ValueError, "probe step not open")
  probeEvent("end step")
  state.insideStep = false
  probeFlush()

proc probeCpuContext*(arena: string): string =
  result = cpuArena
  cpuArena = arena

proc probeCpuRestore*(previous: string) = cpuArena = previous

proc probeVmLaunch*(code: openArray[uint32]; base: pointer; memLen, inputLen, nThreads, tgFloats, tgSize: int; inputBase: pointer = nil) =
  probeLocked()
  let arena = if cpuArena.len > 0: cpuArena else:
    probeBuffer(probeIdentity(base), memLen*4, "arena")
  let identity = probeIdentity(base)
  if cpuArena.len == 0 and not state.vmArenas.hasKey(identity):
    state.vmArenas[identity] = true
    probeEvent("upload t" & $state.nextTransfer & " dst=" & arena & "@0:" & $(memLen*4) &
      probeCapture(base, memLen*4) & " op=" & (if currentOp < 0: "none" else: "o" & $currentOp))
    inc state.nextTransfer
  let program = probeProgram(code)
  var bindings = "bind=(mem:" & arena & "@0:" & $(memLen*4)
  if inputLen > 0:
    let inputId = probeBuffer(probeIdentity(inputBase), inputLen*4, "upload")
    probeEvent("upload t" & $state.nextTransfer & " dst=" & inputId & "@0:" & $(inputLen*4) &
      probeCapture(inputBase, inputLen*4) & " op=" & (if currentOp < 0: "none" else: "o" & $currentOp))
    inc state.nextTransfer
    bindings.add ",input:" & inputId & "@0:" & $(inputLen*4)
  bindings.add ')'
  probeEvent("disp d" & $state.nextDisp & " prog=" & program & " nThreads=" & $nThreads &
    " tgSize=" & $tgSize & " tgFloats=" & $tgFloats & " tgbytes=" & $(tgFloats*4) &
    " mem=" & $memLen & " " & bindings &
    " input=" & $inputLen & " op=" & (if currentOp < 0: "none" else: "o" & $currentOp))
  inc state.nextDisp

proc probeTransfer*(kind, args: string) =
  probeLocked()
  probeEvent(kind & " t" & $state.nextTransfer & args & " op=" & (if currentOp < 0: "none" else: "o" & $currentOp))
  inc state.nextTransfer

proc probeUploaded*(raw, data: pointer; bytes: int) =
  ## Driver uploads also cover callers that bypass the semantic SessionHandle API.
  if not firProbeOn or raw == nil or bytes <= 0: return
  probeLocked()
  let identity = probeNewIdentity()
  probeAlias(raw, identity)
  state.driverUploads[identity] = true
  let id = probeBuffer(identity, bytes, "upload")
  if capturing():
    var small = ""
    if data != nil and bytes mod 4 == 0 and bytes <= probeSmallLimit*4:
      small = probeSmallData(toOpenArray(cast[ptr UncheckedArray[uint32]](data), 0, bytes div 4-1))
    probeTransfer("upload", " dst=" & id & "@0:" & $bytes & small & probeCapture(data, bytes))

proc probeCpuView*(base: pointer; elements: int): string =
  ## CPU multi-buffer launches retain allocation identity and subrange aliases.
  probeLocked()
  let identity = probeIdentity(base)
  result = probeBuffer(identity, elements*4, "arena")
  if not state.vmArenas.hasKey(identity):
    state.vmArenas[identity] = true
    if capturing():
      probeTransfer("upload", " dst=" & result & "@0:" & $(elements*4) & probeCapture(base, elements*4))

proc probeDriverRecorded*(identity: uint64): bool =
  ## The driver hook already emitted the transfer for this upload.
  if not firProbeOn or state == nil: return false
  probeLocked()
  state.driverUploads.hasKey(identity)

proc probeValue*(x: float32): string = probeFloat(x)
proc probeValue*(x: float64): string = probeFloat(x)
proc probeValue*(x: string): string = probeQuote(x)
proc probeValue*(x: pointer): string = (if x == nil: "none" else: "opaque")
proc probeValue*[T](x: openArray[T]): string = "(len=" & $x.len & ")"
proc probeValue*[T](x: T): string =
  when T is SomeInteger or T is bool or T is enum: $x
  else: "opaque"

proc probePrepare*[T](x: var T; origin = "observed") = discard
proc probeRefresh*[T](x: T) = discard

proc probeTypeName(t: NimNode): string =
  var node = t
  if node.kind == nnkVarTy: node = node[0]
  node.repr.replace(" ", "")

proc probeIsHandle(typ: string): bool =
  typ.contains("SessionHandle") or typ.contains("ResidentWeightHandle")

macro probed*(fn: untyped): untyped =
  ## Wraps a session proc with one canonical `op` record. The original body is
  ## expanded unchanged in both branches; metadata exists only when the
  ## recorder is on.
  result = fn.copyNimTree
  let body = fn.body
  let apiLit = newLit($fn.name)
  var before = newStmtList()
  var after = newStmtList()
  var insStr = newLit("")
  var attrsStr = newLit("")
  var outsStr = newLit("")
  var progStr = newLit("none")
  var implStr = newLit("session:" & $fn.name)
  var opParamName = ""
  for group in fn.params[1 .. ^1]:
    let typNode = group[^2]
    let typ = probeTypeName(typNode)
    for nameIdx in 0 ..< group.len-2:
      let name = group[nameIdx]
      let nameS = $name
      if typ == "GpuBackend":
        before.add newCall(ident"probeBackendValue", name)
      elif nameS == "source" or nameS == "glslSrc":
        discard
      elif nameS == "code" and probeIsHandle(typ):
        progStr = newCall(ident"probeCodeProgram", name)
      elif nameS == "kernelName" and typ == "string":
        progStr = newCall(ident"probePipelineProgram", name)
        attrsStr = infix(attrsStr, "&", infix(newLit("kernelName="), "&", newCall(ident"probeValue", name)))
      elif probeIsHandle(typ):
        if typ.contains("ResidentWeightHandle"):
          insStr = infix(insStr, "&", infix(newLit("," & nameS & "="),
            "&", newCall(ident"probeWeightRef", name)))
          attrsStr = infix(attrsStr, "&", newCall(ident"probeWeightAttrs", name))
        elif typNode.kind == nnkVarTy:
          before.add newCall(ident"probePrepare", name)
          after.add newCall(ident"probeRefresh", name)
          outsStr = infix(outsStr, "&", infix(newLit("," & nameS & "="),
            "&", newCall(ident"probeValue", name)))
        else:
          insStr = infix(insStr, "&", infix(newLit("," & nameS & "="),
            "&", newCall(ident"probeValue", name)))
      else:
        attrsStr = infix(attrsStr, "&", infix(newLit("," & nameS & "="),
          "&", newCall(ident"probeValue", name)))
        if nameS == "op" and typ == "string": opParamName = nameS
  let apiName = $fn.name
  if apiName in ["sessionMicroVmEnc", "sessionJitMicroVmEnc"]:
    implStr = newCall(ident"probeProgramImpl", progStr)
  if opParamName.len > 0:
    # Dynamic ATen name (sessionVecOp/sessionActivation carry `op`).
    let opIdent = ident(opParamName)
    let tagCall = newCall(ident"probeTagFor", apiLit, opIdent)
    let ctx = genSym(nskLet, "probeCall")
    let prevArena = genSym(nskLet, "probeArena")
    var arenaExpr: NimNode = newLit("")
    if apiName in ["sessionMicroVmEnc", "sessionJitMicroVmEnc"]:
      arenaExpr = newCall(ident"probeBufferRef", ident"mem")
    result.body = quote do:
      if unlikely(firProbeOn):
        `before`
        let `ctx` = probeOpStart(`apiLit`, `tagCall`, `insStr`, `attrsStr`, `implStr`, `progStr`)
        let `prevArena` = probeCpuContext(`arenaExpr`)
        try:
          `body`
        finally:
          `after`
          probeOpFinish(`ctx`, `outsStr`, "", "")
          probeCpuRestore(`prevArena`)
      else:
        `body`
  else:
    let ctx = genSym(nskLet, "probeCall")
    let prevArena = genSym(nskLet, "probeArena")
    var arenaExpr: NimNode = newLit("")
    if apiName in ["sessionMicroVmEnc", "sessionJitMicroVmEnc"]:
      arenaExpr = newCall(ident"probeBufferRef", ident"mem")
    var post = newStmtList()
    let bytesId = newIdentNode("bytes")
    let dataId = newIdentNode("data")
    let nId = newIdentNode("n")
    let nThreadsId = newIdentNode("nThreads")
    let tgSizeId = newIdentNode("tgSize")
    let tgFloatsId = newIdentNode("tgFloats")
    if apiName == "sessionUploadPrivatePtr":
      post.add quote do:
        if result.kind == gbCpu: probePrepare(result, "upload")
        probeBytesResult(result, `bytesId`)
        if result.kind == gbCpu or not probeUploadRecorded(result):
          probeTransfer("upload", " dst=" & probeValue(result) & probeCapture(`dataId`, `bytesId`))
    elif apiName.startsWith("sessionUpload"):
      post.add quote do:
        if result.kind == gbCpu: probePrepare(result, "upload")
        probeBytesResult(result, `dataId`.len*4)
        if result.kind == gbCpu or not probeUploadRecorded(result):
          probeTransfer("upload", " dst=" & probeValue(result) & probeSmallData(`dataId`) & probeCapture(`dataId`))
    if apiName == "sessionAllocScratch":
      post.add quote do:
        probePrepare(result, "alloc")
        probeBytesResult(result, `nId`*4)
    if apiName == "sessionAllocFromHeap":
      post.add quote do:
        probePrepare(result, "alloc")
        probeBytesResult(result, `nId`*4)
    if apiName in ["sessionMicroVmEnc", "sessionJitMicroVmEnc"]:
      post.add quote do:
        if result and mem.kind != gbCpu:
          probeGpuDispatch(`progStr`, "", `nThreadsId`, `tgSizeId`,
            (if `tgFloatsId` > 0: `tgFloatsId`*4 else: 0),
            "bind=(mem:" & probeHandleRef(mem) & ")")
    if apiName == "sessionCopyInto" and fn.params.len > 5:
      let src = ident"src"; let dst = ident"dst"; let n = ident"n"
      let srcOffset = ident"srcOffsetElems"; let dstOffset = ident"dstOffsetElems"
      post.add quote do:
        probeTransfer("copy", " src=" & probeBufferRef(`src`) & "@" & $(`srcOffset`*4) &
          ":" & $(`n`*4) & " dst=" & probeBufferRef(`dst`) & "@" & $(`dstOffset`*4) &
          ":" & $(`n`*4) & " bytes=" & $(`n`*4) & " ok=" & $result)
    if apiName == "sessionRead":
      let h = ident"h"; let n = ident"n"
      post.add quote do:
        probeTransfer("read", " src=" & probeBufferRef(`h`) & "@0:" & $(`n`*4) & " bytes=" & $(`n`*4))
    if apiName in ["sessionEnd", "sessionCommitContinue"]:
      post.add quote do: probeEvent("barrier kind=host_wait op=" & (if currentOp < 0: "none" else: "o" & $currentOp))
    result.body = quote do:
      if unlikely(firProbeOn):
        `before`
        let `ctx` = probeOpStart(`apiLit`, probeTagFor(`apiLit`), `insStr`, `attrsStr`, `implStr`, `progStr`)
        let `prevArena` = probeCpuContext(`arenaExpr`)
        try:
          `body`
        finally:
          `after`
          `post`
          probeOpFinish(`ctx`, `outsStr`, "", "")
          probeCpuRestore(`prevArena`)
      else:
        `body`

proc probeIsHandleResult*(s: string): bool = s.startsWith("b")

let initialProbePath = getEnv("BYBY_FIR_PROBE")
let initialProbeSteps = getEnv("BYBY_FIR_PROBE_STEPS")
if initialProbePath.len > 0: firProbeStart(initialProbePath, initialProbeSteps)
