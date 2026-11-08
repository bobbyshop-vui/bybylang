## Header-only tensor index for raw checkpoint buffer identities
## (NIMQ2/3, NIMBIN raw fp32, or safetensors).
import std/[os, json, tables, strutils]
import ./probe_data

type ProbeTensor* = object
  name*: string
  shape*: seq[int]
  offset*, bytes*, kind*, scaleOffset*, scales*, groupSize*, eb*, mb*: int
  source*: string

proc readI32(f: File): int =
  var v: int32
  if f.readBuffer(addr v, 4) != 4: raise newException(IOError, "truncated checkpoint header")
  int(v)

proc readI64(f: File): int =
  var v: int64
  if f.readBuffer(addr v, 8) != 8: raise newException(IOError, "truncated checkpoint header")
  if v < 0 or v > int64(high(int)): raise newException(IOError, "invalid checkpoint element count")
  int(v)

proc readString(f: File): string =
  let n = readI32(f)
  if n < 0 or n > 1 shl 24: raise newException(IOError, "invalid checkpoint string length")
  result = newString(n)
  if n > 0 and f.readBuffer(addr result[0], n) != n: raise newException(IOError, "truncated checkpoint string")

proc probeTensorIndex*(path: string): seq[ProbeTensor] =
  var f = open(path, fmRead)
  defer: f.close()
  let fileSize = getFileSize(f)
  if path.endsWith(".safetensors"):
    var headerBytes: uint64
    if f.readBuffer(addr headerBytes, 8) != 8 or headerBytes > 1'u64 shl 28:
      raise newException(IOError, "invalid safetensors header")
    var headerText = newString(int(headerBytes))
    if f.readBuffer(addr headerText[0], headerText.len) != headerText.len:
      raise newException(IOError, "truncated safetensors header")
    let header = parseJson(headerText)
    for name, item in header:
      if name == "__metadata__": continue
      var t = ProbeTensor(name: name, kind: -1, source: path,
        offset: 8+int(headerBytes)+item["data_offsets"][0].getInt(),
        bytes: item["data_offsets"][1].getInt()-item["data_offsets"][0].getInt())
      for dim in item["shape"]: t.shape.add dim.getInt()
      if item["dtype"].getStr() == "F32": t.kind = 0
      result.add t
  else:
    var prefix = newString(8)
    if f.readBuffer(addr prefix[0], 8) != 8: raise newException(IOError, "truncated checkpoint header")
    f.setFilePos(0)
    if prefix.startsWith("NIMBIN"):
      # Raw fp32 state dict: magic, meta pairs, tensors (name, shape, count, fp32 payload).
      f.setFilePos(8)
      let metaCount = readI32(f)
      if metaCount < 0 or metaCount > 1_000_000: raise newException(IOError, "invalid .bin metadata")
      for i in 0 ..< metaCount: discard readString(f); discard readString(f)
      let count = readI32(f)
      if count < 0 or count > 1_000_000: raise newException(IOError, "invalid tensor count")
      for i in 0 ..< count:
        var t = ProbeTensor(name: readString(f), kind: 0, source: path)
        let rank = readI32(f)
        if rank < 0 or rank > 8: raise newException(IOError, "invalid tensor rank")
        for d in 0 ..< rank: t.shape.add readI32(f)
        let elems = readI64(f)
        if elems > int(fileSize div 4): raise newException(IOError, "invalid .bin tensor element count")
        t.bytes = elems*4
        t.offset = int(f.getFilePos())
        f.setFilePos(f.getFilePos()+t.bytes)
        result.add t
    else:
      let magic = readString(f)
      var alignment = 0
      if magic == "NIMQ3":
        alignment = readI32(f)
        if alignment < 64 or alignment > 1 shl 24: raise newException(IOError, "invalid NIMQ3 alignment")
        f.setFilePos(alignment)
      elif magic != "NIMQ2": raise newException(IOError, "unsupported checkpoint format: " & magic)
      for i in 0 ..< 5: discard readI32(f)
      let count = readI32(f)
      if count < 0 or count > 1_000_000: raise newException(IOError, "invalid tensor count")
      for i in 0 ..< count:
        var t = ProbeTensor(name: readString(f), source: path)
        var kind: uint8
        if f.readBuffer(addr kind, 1) != 1: raise newException(IOError, "truncated tensor kind")
        t.kind = int(kind)
        let rank = readI32(f)
        if rank < 0 or rank > 8: raise newException(IOError, "invalid tensor rank")
        for d in 0 ..< rank: t.shape.add readI32(f)
        t.scales = readI32(f)
        if t.scales < 0: raise newException(IOError, "invalid tensor scale count")
        t.scaleOffset = int(f.getFilePos())
        f.setFilePos(f.getFilePos()+int64(t.scales)*8)
        t.groupSize = readI32(f)
        t.eb = readI32(f)
        t.mb = readI32(f)
        t.bytes = readI32(f)
        if alignment > 0: f.setFilePos(f.getFilePos() + (alignment-f.getFilePos() mod alignment) mod alignment)
        t.offset = int(f.getFilePos())
        f.setFilePos(f.getFilePos()+t.bytes)
        result.add t
  for t in result:
    if t.offset < 0 or t.bytes < 0 or int64(t.offset)+t.bytes > fileSize:
      raise newException(IOError, "checkpoint tensor outside file: " & t.name)

proc matchProbeTensor*(index: seq[ProbeTensor]; path: string; bytes: int; fingerprint: string): int =
  ## Ambiguity fails closed. Size/order alone cannot identify permuted equal-sized weights.
  result = -1
  for i, tensor in index:
    if tensor.bytes != bytes: continue
    let source = if tensor.source.len > 0: tensor.source else: path
    if fingerprint.len > 0 and fingerprintFile(source, tensor.offset, bytes) != fingerprint: continue
    if result >= 0: return -2
    result = i
