"""Parse read commands; never forward model-authored shell text to a shell."""

import re
import shlex
from pathlib import PurePosixPath

from examples.seta.p2o.reflection_files import OUTPUT, ROOT


class ReadOnlyCommands:
    def __init__(self, filenames):
        self.files = {ROOT + "/" + name for name in filenames} | {OUTPUT}
        self.directories = {
            str(parent)
            for file in self.files
            for parent in PurePosixPath(file).parents
            if str(parent).startswith(ROOT)
        }

    def _file(self, name):
        absolute = name if name.startswith("/") else ROOT + "/" + name
        if absolute not in self.files:
            raise ValueError("Only files listed in evidence/manifest.json and output/template.md may be read")
        return absolute

    def translate(self, command):
        if len(command) > 65536 or any(ord(c) < 32 and c not in "\n\t" for c in command):
            raise ValueError("Invalid command size or terminal control character")
        parts = shlex.split(command)
        if not parts:
            return "true"
        op, *args = parts
        if op == "write_prompt" and len(args) == 1 and args[0].strip():
            return f"printf %s {shlex.quote(args[0])} > {shlex.quote(OUTPUT)}"
        if op == "ls":
            while args and args[0] in {"-l", "-a", "-la", "-al"}:
                args.pop(0)
            if len(args) > 1:
                raise ValueError("ls accepts one evidence directory")
            name = args[0] if args else ROOT
            path = (name if name.startswith("/") else ROOT + "/" + name).rstrip("/")
            if path not in self.directories:
                raise ValueError("Unknown evidence directory")
            return shlex.join(["ls", "-la", "--", path])
        if op in {"cat", "wc"} and args:
            return shlex.join([op, "--", *(self._file(a) for a in args)])
        if op in {"head", "tail"}:
            count = "100"
            if len(args) == 3 and args[0] == "-n" and args[1].isdigit() and 1 <= int(args[1]) <= 10000:
                count, args = args[1], args[2:]
            if len(args) == 1:
                return shlex.join([op, "-n", count, "--", self._file(args[0])])
        if op == "sed" and len(args) == 3 and args[0] == "-n" and re.fullmatch(r"\d+(,\d+)?p", args[1]):
            return shlex.join([op, "-n", args[1], "--", self._file(args[2])])
        if op == "grep":
            flags = []
            while args and args[0] in {"-n", "-i", "-E", "-F", "-c"}:
                flags.append(args.pop(0))
            if len(args) >= 2:
                return shlex.join([op, *flags, "--", args[0], *(self._file(a) for a in args[1:])])
        raise ValueError(
            "Allowed: ls, cat, head/tail [-n N], sed -n N,Mp, grep [-n -i -E -F -c], wc; write_prompt 'text'"
        )
