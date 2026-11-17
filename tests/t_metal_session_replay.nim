## Metal session record / replay: records a session of a few dispatches (buffer copies, vector ops
## and a JIT kernel with setBytes constants and threadgroup memory) that reads a host-visible input
## buffer and writes a host-visible output, replays it with different inputs, and compares every
## replay against the same session encoded directly and against the CPU. Both replay paths are
## checked: the indirect command buffer and the plain encoder (BYBY_METAL_REPLAY_ICB=0).
## Run: nim c -r -d:release tests/t_metal_session_replay.nim   (macOS with a Metal GPU)

import std/[os, math, strutils]
import ../gpubackend
import ../micro_kernel/session_vm_dispatch

const n = 300                     # not a multiple of the threadgroup width: partial groups too
const tgWidth = 64

const axpySource = """
#include <metal_stdlib>
using namespace metal;
kernel void t_replay_axpy(device float* y [[buffer(0)]], device const float* x [[buffer(1)]],
                          constant int& n [[buffer(2)]], constant float& a [[buffer(3)]],
                          threadgroup float* tg [[threadgroup(0)]],
                          uint i [[thread_position_in_grid]], uint l [[thread_position_in_threadgroup]]) {
    tg[l] = (i < (uint)n) ? x[i] : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (i < (uint)n) y[i] = a * tg[l] + y[i];
}
"""

proc check(cond: bool, what: string) =
  if not cond:
    echo "FAIL: ", what
    quit(1)

proc expected(input, w: seq[float32]): seq[float32] =
  result = newSeq[float32](n)
  for i in 0 ..< n:
    let y = input[i] * w[i] + 2.5'f32 * input[i]
    result[i] = y + w[i]

when not defined(macosx):
  echo "SKIP: Metal only"
  quit(0)

let be = gbMetal
check(ensureGpuInit(be), "Metal init")
check(sessionCanRecord(be), "sessionCanRecord(gbMetal)")
check(sessionMetalJitCompile("t_replay_axpy", axpySource), "JIT compile")

var wData = newSeq[float32](n)
for i in 0 ..< n: wData[i] = float32(i mod 7) * 0.25'f32 - 0.5'f32
var w = sessionUpload(be, wData)
var x = sessionAllocScratch(be, n)
var y = sessionAllocScratch(be, n)
var z = sessionAllocScratch(be, n)
var hostIn = sessionHostBuffer(be, n)
var hostOut = sessionHostBuffer(be, n, true)
check(hostIn.metalHandle != nil and hostOut.metalHandle != nil, "host buffers")

proc encodeStep() =
  ## Four dispatch kinds in one open session: copy in, mul, JIT axpy, add, copy out.
  check(sessionCopyInto(be, hostIn, x, 0, 0, n), "copy in")
  check(sessionVecOp(be, "mul", x, w, y, n), "mul")
  check(sessionCommitContinue(be), "mid-session commit")   # a split of the recording
  check(sessionEncPipeline(false, "t_replay_axpy"), "axpy pipeline")
  check(sessionBindBuffer(0, y.metalHandle) and sessionBindBuffer(1, x.metalHandle), "axpy bind")
  check(sessionBindInt(2, n) and sessionBindFloat(3, 2.5'f32), "axpy constants")
  check(sessionSetThreadgroupBytes(0, tgWidth*4), "axpy threadgroup memory")
  check(sessionDispatch(n, tgWidth), "axpy dispatch")
  check(sessionVecOp(be, "add", y, w, z, n), "add")
  check(sessionCopyInto(be, z, hostOut, 0, 0, n), "copy out")

proc inputFor(trial: int): seq[float32] =
  result = newSeq[float32](n)
  for i in 0 ..< n: result[i] = sin(float32(i + 1) * 0.37'f32 + float32(trial)) * float32(trial + 1)

proc direct(input: seq[float32]): seq[float32] =
  check(sessionHostWrite(be, hostIn, input), "host write")
  check(sessionBegin(be), "direct begin")
  encodeStep()
  check(sessionEnd(be), "direct end")
  sessionHostRead(be, hostOut, n)

proc record(input: seq[float32]): int =
  check(sessionHostWrite(be, hostIn, input), "host write")
  check(sessionBeginRecording(be), "begin recording")
  encodeStep()
  check(sessionHostReadBarrier(be), "host read barrier")
  check(sessionEnd(be), "recording end")
  let got = sessionHostRead(be, hostOut, n)
  let want = expected(input, wData)
  for i in 0 ..< n: check(abs(got[i] - want[i]) <= 1e-4'f32 * max(1'f32, abs(want[i])), "recorded step value " & $i)
  sessionLastRecording(be)

var failures = 0
for mode in ["0", "1"]:
  putEnv("BYBY_METAL_REPLAY_ICB", mode)
  let id = record(inputFor(0))
  check(id >= 0, "recording kept (mode " & mode & "): " & sessionRecordingNote(be, id))
  let note = sessionRecordingNote(be, id)
  echo "mode BYBY_METAL_REPLAY_ICB=", mode, ": recording ", id, " replays on ", note
  if mode == "0": check(note == "plain encoder", "plain path chosen")
  for trial in 1 .. 5:
    let input = inputFor(trial)
    let want = direct(input)
    let cpu = expected(input, wData)
    check(sessionHostWrite(be, hostIn, inputFor(trial + 100)), "scribble")   # replay must read the input written next
    check(sessionHostWrite(be, hostIn, input), "host write")
    check(sessionReplay(be, id), "replay")
    let got = sessionHostRead(be, hostOut, n)
    var worst = 0'f32
    for i in 0 ..< n:
      if got[i] != want[i]: inc failures
      worst = max(worst, abs(got[i] - cpu[i]))
    check(worst <= 1e-4'f32 * 10, "replay vs CPU, trial " & $trial)
    echo "  trial ", trial, ": replay == direct encode (", n, " values), max |replay - cpu| = ", formatFloat(worst, ffScientific, 2)
  sessionFreeRecording(be, id)
check(failures == 0, $failures & " replayed values differ from the direct encode")

# A session that uploads while recording is not kept.
check(sessionBeginRecording(be), "begin recording (refusal)")
var staged = sessionUpload(be, inputFor(7))
check(sessionCopyInto(be, staged, x, 0, 0, n), "staged copy")
check(sessionEnd(be), "refusal end")
let refused = sessionLastRecording(be)
check(refused < 0, "a session with an upload must not be kept")
echo "upload while recording: not kept (", sessionRecordingNote(be, refused), ")"
sessionFree(be, staged)

for h in [w, x, y, z, hostIn, hostOut]:
  var t = h
  sessionFree(be, t)
echo "PASS"
