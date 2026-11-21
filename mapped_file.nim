## Small POSIX memory mapping wrapper for long-lived model checkpoint ranges.
import std/memfiles
import ./probe

type MappedReadOnlyFile* = object
  data*: pointer
  size*: int
  backing: MemFile

proc openReadOnlyMapping*(path: string): MappedReadOnlyFile  {.apiHooked.}=
  result.backing = memfiles.open(path, fmRead)
  result.data = result.backing.mem
  result.size = result.backing.size

proc unmapReadOnlyFile*(mapping: var MappedReadOnlyFile)  {.apiHooked.}=
  if mapping.data != nil:
    memfiles.close(mapping.backing)
    mapping.data = nil
    mapping.size = 0
