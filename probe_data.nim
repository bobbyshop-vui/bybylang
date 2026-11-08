## Versioned sampled content identity. Only 5*64 bytes are inspected per upload.
import std/[strutils, os]

proc fingerprintOffsets*(bytes: int): seq[int] =
  if bytes <= 0: return
  let width = min(64, bytes)
  for i in 0 .. 4:
    let off = (bytes-width)*i div 4
    if off notin result: result.add off

proc fingerprint*(data: pointer; bytes: int): string =
  var hash = 0xcbf29ce484222325'u64
  for i in 0 .. 7: hash = (hash xor uint64((uint64(bytes) shr (8*i)) and 255)) * 0x100000001b3'u64
  let p = cast[ptr UncheckedArray[uint8]](data)
  for off in fingerprintOffsets(bytes):
    for j in 0 ..< min(64, bytes): hash = (hash xor uint64(p[off+j])) * 0x100000001b3'u64
  "sample1-" & toHex(hash, 16).toLowerAscii

proc fingerprintFile*(path: string; offset, bytes: int): string =
  var f = open(path, fmRead)
  defer: f.close()
  if offset < 0 or bytes < 0 or int64(offset)+int64(bytes) > getFileSize(f):
    raise newException(IOError, "fingerprint range outside file: " & path)
  # Keep the same byte stream as fingerprint(pointer), without allocating a tensor.
  var hash = 0xcbf29ce484222325'u64
  for i in 0 .. 7: hash = (hash xor uint64((uint64(bytes) shr (8*i)) and 255)) * 0x100000001b3'u64
  var blockData: array[64, uint8]
  for off in fingerprintOffsets(bytes):
    f.setFilePos(offset+off)
    let width = min(64, bytes)
    if f.readBuffer(addr blockData[0], width) != width: raise newException(IOError, "truncated fingerprint data")
    for j in 0 ..< width: hash = (hash xor uint64(blockData[j])) * 0x100000001b3'u64
  "sample1-" & toHex(hash, 16).toLowerAscii
