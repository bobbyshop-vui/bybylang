import strutils, sequtils
import gpubackend
const RAM_SIZE = 1024
var RAM: array[0..RAM_SIZE-1, int]
var BUS: seq[string] = @[]
var Pins: array[0..31, bool]

proc stripQuotes(s: string): string =
  if s.len >= 2 and s[0] == '"' and s[^1] == '"':
    return s[1..^2]
  else:
    return s

proc apuTran(name: string, payload: string) =
  BUS.add(payload)
  echo "[APU-TRAN] ", name, " -> ", payload

proc apuMem(action: string, target: string, value: string) =
  var ramAddr = parseInt(target.replace("RAM", ""))
  if action == "write":
    RAM[ramAddr] = parseInt(value)
  elif action == "read":
    echo "[APU-MEM] RAM[", ramAddr, "] -> ", RAM[ramAddr]

proc apuCore(mode: int, code: string) =
  echo "[APU-CORE] Mode:", mode, " run:", code

proc apuPin(pin: int, state: string) =
  Pins[pin] = (state == "high")

proc bitSend(bits: string) =
  BUS.add(bits)

proc bitRecv() =
  if BUS.len > 0:
    echo BUS[0]
    delete(BUS, 0)
  else:
    echo "[BIT-RECV] empty"

proc memMap(target: string) =
  echo "[MEM-MAP] ", target

proc memPush(target: string, value: string) =
  echo "[MEM-PUSH] ", target, " <- ", value

proc tranPulse(pin: int, width: string) =
  echo "[TRAN-PULSE] pin ", pin, " width ", width


echo "Mode 3: High-level"
gpuBackendSelected = parseBackend("metal")
var A: seq[float32] = @[1, 2, 3, 4, 5, 6, 7, 8].mapIt(it.float32)
var B: seq[float32] = @[10, 20, 30, 40, 50, 60, 70, 80].mapIt(it.float32)
var C = gpuOp("add", gpuBackendSelected, A, B)
var D = gpuOp("mul", gpuBackendSelected, A, B)
echo "GPU add result:"
echo C
echo "GPU mul result:"
echo D
var x = 5
if x > 0:
  echo "x duong"
  if x > 3:
    echo "x lon hon 3"
    if x == 5:
      echo "x dung bang 5 - if long 3 cap hoat dong!"
    else:
      echo "x khac 5"
  else:
    echo "x tu 1 den 3"
elif x == 0:
  echo "x bang 0"
else:
  echo "x am"
for i in 1..4:
  if i == 2:
    echo "gap so 2 trong vong lap"
  else:
    echo "khong phai so 2"