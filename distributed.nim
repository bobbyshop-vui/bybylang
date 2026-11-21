## Native process scheduling for data-parallel workers.
## Each child receives its selected backend and device ordinal before it starts.

import std/[os, osproc, strtabs, strutils, times, tables, math, sequtils, streams]
import ./probe
when defined(posix):
  import std/posix

type
  ParallelDevice* = object
    backend*: string
    ordinal*: int
  ParallelCommand* = object
    executable*: string
    args*: seq[string]
    environment*: seq[(string, string)]
    unsetEnvironment*: seq[string]
    input*: string
    replaceBackendArg*: bool
    backendArgIndex*: int
  ParallelResult* = object
    stdout*, stderr*: string
  ProcessGroup* = ref object
    devices*: seq[ParallelDevice]
    initialRatios: seq[float64]
    fixedRatios: bool
    rates: Table[string, seq[float64]]

  RunningChild = object
    process: Process
    rank, resultIndex: int
    started: float
    outOpen, errOpen: bool
    inOpen: bool
    input: string
    inputOffset: int

proc normalizedRatios(group: ProcessGroup; requested: openArray[float64]): seq[float64]

proc initProcessGroup*(devices: openArray[string]; ratios: openArray[float64] = []): ProcessGroup {.apiHooked.} =
  if devices.len == 0:
    raise newException(ValueError, "A process group requires at least one device")
  if ratios.len > 0 and ratios.len != devices.len:
    raise newException(ValueError, "Ratio count must match device count")
  result = ProcessGroup(devices: @[], rates: initTable[string, seq[float64]]())
  var seen = initTable[string, bool]()
  for spec in devices:
    let split = spec.rfind(':')
    if split >= 0 and split == spec.high:
      raise newException(ValueError, "Missing device ordinal in " & spec)
    let backend = (if split <= 0: spec else: spec[0 ..< split]).toLowerAscii
    let ordinalText = if split <= 0: "0" else: spec[split + 1 .. ^1]
    if backend notin ["cpu", "vulkan", "metal"]:
      raise newException(ValueError, "Unknown device backend: " & backend)
    var ordinal: int
    try:
      ordinal = parseInt(ordinalText)
    except ValueError:
      raise newException(ValueError, "Invalid device ordinal in " & spec)
    if backend.len == 0 or ordinal < 0 or (backend == "cpu" and ordinal != 0):
      raise newException(ValueError, "Invalid device: " & spec)
    let identity = backend & ":" & $ordinal
    if seen.hasKey(identity):
      raise newException(ValueError, "Duplicate device: " & identity)
    seen[identity] = true
    result.devices.add ParallelDevice(backend: backend, ordinal: ordinal)
  if ratios.len > 0:
    result.initialRatios = normalizedRatios(result, ratios)
    result.fixedRatios = true

proc initialRates(group: ProcessGroup): seq[float64] =
  for d in group.devices:
    result.add(if d.backend == "cpu": 1.0 else: 8.0)

proc capCpuBelowGpu(group: ProcessGroup; rates: var seq[float64]) =
  var slowestGpu = Inf
  var cpuIndex = -1
  for i, device in group.devices:
    if device.backend == "cpu": cpuIndex = i
    else: slowestGpu = min(slowestGpu, rates[i])
  if cpuIndex >= 0 and slowestGpu < Inf:
    rates[cpuIndex] = min(rates[cpuIndex], max(0.01, slowestGpu * 0.5))

proc normalizedRatios(group: ProcessGroup; requested: openArray[float64]): seq[float64] =
  if requested.len == 0:
    return
  if requested.len != group.devices.len:
    raise newException(ValueError, "Ratio count must match device count")
  var total = 0.0
  for x in requested:
    if x < 0 or classify(x) in {fcNan, fcInf, fcNegInf}:
      raise newException(ValueError, "Ratios must be finite and non-negative")
    total += x
    result.add(x)
  if total <= 0:
    raise newException(ValueError, "At least one ratio must be positive")
  if classify(total) in {fcNan, fcInf, fcNegInf}:
    raise newException(ValueError, "Ratio total must be finite")

proc weights(group: ProcessGroup; workload: string; requested: openArray[float64]): seq[float64] =
  if requested.len > 0:
    return normalizedRatios(group, requested)
  if group.rates.hasKey(workload):
    return group.rates[workload]
  if group.initialRatios.len > 0:
    return group.initialRatios
  result = initialRates(group)

proc partition*(group: ProcessGroup; length: int; ratios: openArray[float64] = []): seq[int] {.apiHooked.} =
  if group.isNil or group.devices.len == 0:
    raise newException(ValueError, "Process group has no devices")
  if length < 0:
    raise newException(ValueError, "Length must be non-negative")
  let ws = if ratios.len > 0: normalizedRatios(group, ratios)
           elif group.initialRatios.len > 0: group.initialRatios
           else: initialRates(group)
  if uint64(length) > 9_007_199_254_740_991'u64:
    raise newException(ValueError, "Length exceeds exact float partition range")
  result = newSeq[int](ws.len)
  let total = ws.foldl(a + b, 0.0)
  var remainders = newSeq[float64](ws.len)
  var assigned = 0
  for i, w in ws:
    let exact = float64(length) * (w / total)
    result[i] = int(floor(exact))
    remainders[i] = exact - float64(result[i])
    assigned += result[i]
  while assigned < length:
    var best = -1
    for i, rem in remainders:
      if ws[i] > 0 and (best < 0 or rem > remainders[best]): best = i
    if best < 0: break
    inc result[best]
    remainders[best] = -1
    inc assigned

proc childEnvironment(command: ParallelCommand; device: ParallelDevice): StringTableRef =
  result = newStringTable(mode = modeCaseSensitive)
  for (k, v) in envPairs(): result[k] = v
  for (k, v) in command.environment: result[k] = v
  for k in command.unsetEnvironment: result.del(k)
  result["BYBY_PARALLEL_BACKEND"] = device.backend
  result["BYBY_PARALLEL_DEVICE"] = $device.ordinal
  if device.backend == "vulkan": result["BYBY_VULKAN_DEVICE"] = $device.ordinal
  elif device.backend == "metal": result["BYBY_METAL_DEVICE"] = $device.ordinal

proc commandArgs(command: ParallelCommand; device: ParallelDevice): seq[string] =
  result = command.args
  if command.replaceBackendArg:
    if command.backendArgIndex < 0 or command.backendArgIndex >= result.len:
      raise newException(ValueError, "backendArgIndex is outside command args")
    result[command.backendArgIndex] = device.backend

when defined(posix):
  proc writeWithoutSigpipe(fd: cint; data: pointer; size: int): int =
    var blocked, previous: Sigset
    discard sigemptyset(blocked)
    discard sigaddset(blocked, SIGPIPE)
    if pthread_sigmask(SIG_BLOCK, blocked, previous) != 0:
      raise newException(IOError, "Unable to block SIGPIPE for worker stdin write")
    var pending: Sigset
    discard sigpending(pending)
    let hadPending = sigismember(pending, SIGPIPE) == 1
    result = posix.write(fd, data, size)
    let writeErrno = errno
    if result < 0 and errno == EPIPE and not hadPending:
      var afterWrite: Sigset
      discard sigpending(afterWrite)
      if sigismember(afterWrite, SIGPIPE) == 1:
        var received: cint
        discard sigwait(blocked, received)
    discard pthread_sigmask(SIG_SETMASK, previous, blocked)
    errno = writeErrno

proc startChild(command: ParallelCommand; device: ParallelDevice; rank, resultIndex: int): RunningChild =
  let args = commandArgs(command, device)
  result.process = startProcess(command.executable, args = args,
    env = childEnvironment(command, device), options = {poUsePath})
  result.rank = rank
  result.resultIndex = resultIndex
  result.started = epochTime()
  result.outOpen = true
  result.errOpen = true
  result.input = command.input
  result.inOpen = true
  try:
    if command.input.len == 0:
      inputStream(result.process).close()
      result.inOpen = false
    else:
      when defined(posix):
        let fd = cint(inputHandle(result.process))
        let flags = posix.fcntl(fd, F_GETFL)
        if flags < 0 or posix.fcntl(fd, F_SETFL, flags or O_NONBLOCK) < 0:
          raise newException(IOError, "Unable to make worker stdin nonblocking")
      else:
        raise newException(IOError, "Binary worker stdin is currently supported on POSIX systems only")
  except:
    try: kill(result.process)
    except: discard
    try: discard waitForExit(result.process)
    except: discard
    try: close(result.process)
    except: discard
    result.process = nil
    raise

proc runAssigned(group: ProcessGroup; commands: openArray[ParallelCommand]; counts: seq[int];
                 results: var seq[ParallelResult]; elapsed: var seq[float64]) =
  when not defined(posix):
    raise newException(IOError, "Native process scheduler is currently supported on POSIX systems only")
  else:
    var child = newSeq[RunningChild](group.devices.len)
    var active = 0
    var cursors = newSeq[int](group.devices.len)
    var offset = 0
    try:
      for rank, count in counts:
        cursors[rank] = offset
        if count > 0:
          child[rank] = startChild(commands[offset], group.devices[rank], rank, offset)
          inc active
        offset += count
      while active > 0:
        var polls: seq[TPollfd] = @[]
        var owners: seq[(int, char)] = @[]
        for i in 0 ..< child.len:
          if child[i].process.isNil: continue
          if child[i].inOpen:
            polls.add TPollfd(fd: cint(inputHandle(child[i].process)), events: POLLOUT or POLLHUP, revents: 0)
            owners.add (i, 'i')
          if child[i].outOpen:
            polls.add TPollfd(fd: cint(outputHandle(child[i].process)), events: POLLIN or POLLHUP, revents: 0)
            owners.add (i, 'o')
          if child[i].errOpen:
            polls.add TPollfd(fd: cint(errorHandle(child[i].process)), events: POLLIN or POLLHUP, revents: 0)
            owners.add (i, 'e')
        if polls.len > 0:
          let pollResult = posix.poll(addr polls[0], Tnfds(polls.len), 100)
          if pollResult < 0:
            if errno == EINTR: continue
            raise newException(IOError, "poll failed while collecting worker output")
          for pi, pfd in polls:
            if pfd.revents == 0: continue
            let (ci, streamKind) = owners[pi]
            if (pfd.revents and POLLNVAL) != 0:
              raise newException(IOError, "poll reported an invalid worker pipe")
            if streamKind == 'i':
              if (pfd.revents and (POLLHUP or POLLERR)) != 0:
                inputStream(child[ci].process).close()
                child[ci].inOpen = false
                if child[ci].inputOffset < child[ci].input.len:
                  raise newException(IOError, "Worker closed stdin before reading its full input")
                continue
              let remaining = child[ci].input.len - child[ci].inputOffset
              let amount = min(remaining, 8192)
              let n = writeWithoutSigpipe(pfd.fd, addr child[ci].input[child[ci].inputOffset], amount)
              if n > 0:
                child[ci].inputOffset += n
                if child[ci].inputOffset >= child[ci].input.len:
                  inputStream(child[ci].process).close()
                  child[ci].inOpen = false
              elif n < 0 and errno != EINTR and errno != EAGAIN and errno != EWOULDBLOCK:
                raise newException(IOError, "Unable to write worker stdin")
            else:
              var buf = newString(8192)
              let n = posix.read(pfd.fd, addr buf[0], buf.len)
              if n > 0:
                buf.setLen(n)
                if streamKind == 'e': results[child[ci].resultIndex].stderr.add(buf)
                else: results[child[ci].resultIndex].stdout.add(buf)
              elif n == 0:
                if streamKind == 'e': child[ci].errOpen = false
                else: child[ci].outOpen = false
              elif errno != EINTR and errno != EAGAIN and errno != EWOULDBLOCK:
                raise newException(IOError, "Unable to read worker output")
        for rank in 0 ..< child.len:
          if child[rank].process.isNil or child[rank].inOpen or child[rank].outOpen or child[rank].errOpen: continue
          let code = waitForExit(child[rank].process)
          let resultIndex = child[rank].resultIndex
          elapsed[resultIndex] = max(0.000001, epochTime() - child[rank].started)
          close(child[rank].process)
          child[rank].process = nil
          dec active
          if code != 0:
            raise newException(IOError, "Parallel worker rank " & $rank & " command " & $resultIndex &
              " exited with status " & $code & "\n" & results[resultIndex].stderr)
          inc cursors[rank]
          let blockEnd = (if rank == 0: 0 else: counts[0 ..< rank].foldl(a + b, 0)) + counts[rank]
          if cursors[rank] < blockEnd:
            child[rank] = startChild(commands[cursors[rank]], group.devices[rank], rank, cursors[rank])
            inc active
    except:
      for i in 0 ..< child.len:
        if not child[i].process.isNil:
          try: kill(child[i].process)
          except: discard
          try: discard waitForExit(child[i].process)
          except: discard
          try: close(child[i].process)
          except: discard
      raise

proc runCommands*(group: ProcessGroup; commands: openArray[ParallelCommand];
                  ratios: openArray[float64] = []; workload = "commands"): seq[ParallelResult] {.apiHooked.} =
  if group.isNil or group.devices.len == 0:
    raise newException(ValueError, "Process group has no devices")
  if ratios.len > 0: discard normalizedRatios(group, ratios)
  if commands.len == 0: return @[]
  let ws = weights(group, workload, ratios)
  let counts = partition(group, commands.len, ws)
  result = newSeq[ParallelResult](commands.len)
  var elapsed = newSeq[float64](commands.len)
  runAssigned(group, commands, counts, result, elapsed)
  if ratios.len == 0 and not group.fixedRatios:
    var offset = 0
    var updated = ws
    for rank, count in counts:
      if count > 0:
        var duration = 0.0
        for i in offset ..< offset + count: duration += elapsed[i]
        let observed = float64(count) / max(duration, 0.000001)
        updated[rank] = max(0.01, 0.75 * ws[rank] + 0.25 * observed)
      offset += count
    capCpuBelowGpu(group, updated)
    group.rates[workload] = updated

proc runShards*(group: ProcessGroup; length: int;
                build: proc(device: ParallelDevice; offset, count: int): ParallelCommand {.closure.};
                ratios: openArray[float64] = []; workload = "shards"): seq[ParallelResult] {.apiHooked.} =
  if group.isNil or group.devices.len == 0:
    raise newException(ValueError, "Process group has no devices")
  if ratios.len > 0: discard normalizedRatios(group, ratios)
  if length < 0: raise newException(ValueError, "Length must be non-negative")
  if length == 0: return @[]
  let ws = weights(group, workload, ratios)
  let counts = partition(group, length, ws)
  result = @[]
  var elapsed = newSeq[float64](group.devices.len)
  var commands = newSeq[ParallelCommand](group.devices.len)
  var offset = 0
  var ranks: seq[int] = @[]
  for rank, count in counts:
    if count > 0:
      commands[rank] = build(group.devices[rank], offset, count)
      ranks.add rank
    offset += count
  var compact = newSeq[ParallelCommand](ranks.len)
  var compactCounts = newSeq[int](ranks.len)
  for i, rank in ranks:
    compact[i] = commands[rank]
    compactCounts[i] = 1
  var compactResults = newSeq[ParallelResult](ranks.len)
  var compactElapsed = newSeq[float64](ranks.len)
  var activeGroup = ProcessGroup(devices: @[])
  for rank in ranks: activeGroup.devices.add group.devices[rank]
  runAssigned(activeGroup, compact, compactCounts, compactResults, compactElapsed)
  if ratios.len == 0 and not group.fixedRatios:
    var updated = ws
    for i, rank in ranks:
      let observed = float64(counts[rank]) / max(compactElapsed[i], 0.000001)
      updated[rank] = max(0.01, 0.75 * ws[rank] + 0.25 * observed)
    capCpuBelowGpu(group, updated)
    group.rates[workload] = updated
  for i, rank in ranks:
    result.add compactResults[i]
    elapsed[rank] = compactElapsed[i]
  if ratios.len == 0 and not group.fixedRatios:
    var updated = ws
    for rank, count in counts:
      if count > 0:
        let observed = float64(count) / max(elapsed[rank], 0.000001)
        updated[rank] = max(0.01, 0.75 * ws[rank] + 0.25 * observed)
    capCpuBelowGpu(group, updated)
    group.rates[workload] = updated
