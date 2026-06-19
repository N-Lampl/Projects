"""A pure-Python pickle-opcode scanner — the offline stand-in for modelscan.

Pickle is a tiny stack VM. The ability to run arbitrary code at load time comes
from a small set of opcodes that import/look-up and then *call* Python objects:

    GLOBAL / STACK_GLOBAL   -> push a named object (e.g. os.system) onto the stack
    REDUCE / INST / OBJ     -> call a callable with arguments (the actual exploit)
    NEWOBJ / BUILD          -> object construction hooks (__reduce__, __setstate__)

We never *unpickle* the file (that would be the very thing we are defending
against). Instead we walk the opcode stream with the stdlib `pickletools`
disassembler — read-only — and flag the dangerous opcodes and the global
references they resolve. This is the same idea modelscan/picklescan use; it runs
with zero extra dependencies so the project always has a working detector.
"""

from __future__ import annotations

import io
import pickletools
from dataclasses import dataclass, field
from pathlib import Path

# Opcodes that can cause code execution during unpickling.
DANGEROUS_OPCODES = {
    "GLOBAL",
    "STACK_GLOBAL",
    "REDUCE",
    "INST",
    "OBJ",
    "NEWOBJ",
    "NEWOBJ_EX",
    "BUILD",
}

# Module.attr references that are almost never legitimate inside a model file.
SUSPICIOUS_GLOBALS = {
    "os.system",
    "os.popen",
    "os.execv",
    "os.execve",
    "posix.system",
    "nt.system",
    "subprocess.Popen",
    "subprocess.call",
    "subprocess.run",
    "subprocess.check_output",
    "builtins.exec",
    "builtins.eval",
    "builtins.compile",
    "builtins.__import__",
    "builtins.getattr",
    "importlib.import_module",
    "runpy._run_code",
    "pty.spawn",
    "socket.socket",
}


@dataclass
class Finding:
    opcode: str
    arg: str | None
    pos: int
    reason: str


@dataclass
class ScanResult:
    path: str
    is_pickle: bool
    findings: list[Finding] = field(default_factory=list)
    globals_seen: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def malicious(self) -> bool:
        """Verdict: any dangerous opcode or known-bad global ⇒ block."""
        return any(f.opcode in DANGEROUS_OPCODES for f in self.findings) or any(
            g in SUSPICIOUS_GLOBALS for g in self.globals_seen
        )

    @property
    def verdict(self) -> str:
        return "MALICIOUS" if self.malicious else "clean"


def _looks_like_pickle(head: bytes) -> bool:
    # Protocol-2+ files start with PROTO (\x80); protocol-0/1 commonly start
    # with one of the framing/global opcodes. This is a cheap pre-filter.
    if not head:
        return False
    return head[0] in (0x80,) or head[:1] in (b"(", b"c", b"]", b"}", b"K")


def scan_pickle_bytes(data: bytes, path: str = "<bytes>") -> ScanResult:
    """Statically disassemble a pickle byte stream and flag risky opcodes."""
    result = ScanResult(path=path, is_pickle=_looks_like_pickle(data[:2]))
    try:
        for opcode, arg, pos in pickletools.genops(io.BytesIO(data)):
            name = opcode.name
            if name in ("GLOBAL", "STACK_GLOBAL"):
                ref = _normalize_global(name, arg)
                if ref:
                    result.globals_seen.append(ref)
                bad = ref in SUSPICIOUS_GLOBALS
                result.findings.append(
                    Finding(
                        opcode=name,
                        arg=ref,
                        pos=pos,
                        reason=(
                            f"resolves global {ref!r} (KNOWN-DANGEROUS)"
                            if bad
                            else f"resolves global {ref!r}"
                        ),
                    )
                )
            elif name in DANGEROUS_OPCODES:
                result.findings.append(
                    Finding(
                        opcode=name,
                        arg=str(arg) if arg is not None else None,
                        pos=pos,
                        reason=f"opcode {name} can trigger code execution on load",
                    )
                )
    except Exception as exc:  # not a valid pickle / truncated stream
        result.error = str(exc)
    return result


def _normalize_global(opcode_name: str, arg) -> str | None:
    """GLOBAL carries 'module attr'; STACK_GLOBAL pops two strings off the stack."""
    if opcode_name == "GLOBAL" and isinstance(arg, str):
        return arg.replace(" ", ".", 1)
    return None  # STACK_GLOBAL args are resolved at runtime; opcode alone flags it


def scan_pickle_file(path: str | Path) -> ScanResult:
    path = Path(path)
    data = path.read_bytes()
    res = scan_pickle_bytes(data, path=str(path))
    return res
