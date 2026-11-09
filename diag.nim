## diag.nim - clang-style diagnostics for BybyLang and for libraries written in Nim.
##
## Two users share this module:
##
##   * the compiler (`bybylang.nim`) renders syntax errors with the offending source line and a caret;
##   * programs and libraries call `warn`, `error` and `traceback` to report problems with a location:
##
##       import diag
##       warn "buffer is larger than the tile"      # file.nim:12:3: warning: buffer is larger ...
##       traceback "unexpected state"               # prints the call stack, then keeps running
##       error "model.fir is truncated"             # prints, then raises BybyError
##
##     In BybyLang source the same three words work as statements (`warn "text"`); the compiler passes
##     the `.bybylang` file and line, so the message points at the BybyLang code, not at generated Nim.
##
## Output goes to stderr. Colors are used only on a terminal and are off when `NO_COLOR` is set.

import std/[os, strutils, terminal]

type
  DiagSeverity* = enum
    dsNote = "note"
    dsWarning = "warning"
    dsError = "error"

  BybyError* = object of CatchableError
    ## Raised by `error`; carries the already printed message.

var
  gDiagWarnings* = 0     ## warnings reported so far in this process
  gDiagErrors* = 0       ## errors reported so far in this process
  gDiagWarningsAsErrors* = false

proc diagUseColor(): bool =
  stderr.isatty and not existsEnv("NO_COLOR")

proc paint(text, code: string): string =
  if diagUseColor(): "\e[" & code & "m" & text & "\e[0m" else: text

proc severityColor(s: DiagSeverity): string =
  case s
  of dsNote: "1;36"
  of dsWarning: "1;35"
  of dsError: "1;31"

proc escapeLiteral*(s: string): string =
  ## `s` as a Nim string literal (used by the compiler when it writes file names into generated code).
  result = "\""
  for c in s:
    case c
    of '"': result.add "\\\""
    of '\\': result.add "\\\\"
    of '\n': result.add "\\n"
    else: result.add c
  result.add '"'

proc renderDiag*(severity: DiagSeverity; file: string; line, col: int; msg: string;
                 source = ""; caretLen = 1; notes: seq[string] = @[]): string =
  ## Format one diagnostic the way clang does:
  ##
  ##   file.bybylang:3:7: error: message
  ##       return a +
  ##                ^
  ##   note: extra detail
  ##
  ## `line`/`col` are 1-based; `source` is the text of that line (empty = do not print it).
  var head = ""
  if file.len > 0:
    head = paint(file & ":" & (if line > 0: $line & ":" else: "") &
                 (if line > 0 and col > 0: $col & ":" else: ""), "1") & " "
  head &= paint($severity & ":", severityColor(severity)) & " " & paint(msg, "1")
  result = head
  if source.len > 0 and col > 0:
    result &= "\n" & source
    let width = max(1, caretLen)
    result &= "\n" & repeat(' ', col - 1) & paint("^" & repeat('~', width - 1), "1;32")
  for n in notes:
    result &= "\n" & paint("note:", severityColor(dsNote)) & " " & n

proc diagEmit*(severity: DiagSeverity; file: string; line, col: int; msg: string;
               source = ""; caretLen = 1; notes: seq[string] = @[]) =
  var sev = severity
  if sev == dsWarning and gDiagWarningsAsErrors: sev = dsError
  case sev
  of dsWarning: inc gDiagWarnings
  of dsError: inc gDiagErrors
  of dsNote: discard
  stderr.writeLine renderDiag(sev, file, line, col, msg, source, caretLen, notes)
  stderr.flushFile()

# --------------------------
# warn / error / traceback
# --------------------------

proc diagWarn*(msg: string; file = ""; line = 0) =
  ## Report a warning at `file:line`. The program keeps running.
  diagEmit(dsWarning, file, line, 0, msg)

proc diagError*(msg: string; file = ""; line = 0) {.noreturn.} =
  ## Report an error at `file:line` and raise `BybyError`.
  diagEmit(dsError, file, line, 0, msg)
  raise newException(BybyError, msg)

proc diagTraceback*(msg = ""; file = ""; line = 0) =
  ## Print the call stack (oldest call first, like Python). The program keeps running.
  ## `msg` is an optional note; `file:line` is where `traceback` was written.
  var text = "traceback (most recent call last):"
  let frames = getStackTraceEntries()
  if frames.len == 0:
    text &= "\n  (no stack frames: build without --stackTrace:off)"
  for f in frames:
    if $f.procname == "diagTraceback": continue
    text &= "\n  " & $f.filename & ":" & $f.line & " in " & $f.procname
  if file.len > 0:
    text &= "\n  " & file & (if line > 0: ":" & $line else: "") & " <- traceback"
  if msg.len > 0:
    text &= "\n" & paint("note:", severityColor(dsNote)) & " " & msg
  stderr.writeLine text
  stderr.flushFile()

# A BybyError has already been printed with its location; do not add Nim's own "unhandled exception"
# report on top. Other uncaught exceptions keep Nim's default text.
onUnhandledException = proc (msg: string) {.gcsafe.} =
  if not msg.contains("[BybyError]"):
    stderr.write msg
    stderr.flushFile()

template warn*(msg: string) =
  ## Library form: reports the Nim file and line of the call.
  let info = instantiationInfo(fullPaths = false)
  diagWarn(msg, info.filename, info.line)

template error*(msg: string) =
  ## Library form: prints the error with the call site, then raises `BybyError`.
  let info = instantiationInfo(fullPaths = false)
  diagError(msg, info.filename, info.line)

template traceback*(msg = "") =
  ## Library form: prints the call stack with the call site. Does not stop the program.
  let info = instantiationInfo(fullPaths = false)
  diagTraceback(msg, info.filename, info.line)
