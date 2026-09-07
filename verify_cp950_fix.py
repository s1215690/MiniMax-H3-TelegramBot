"""Verify the cp950/UnicodeEncodeError fix WITHOUT launching ComfyUI.

1. Demonstrates the original crash: strict cp950 TextIOWrapper + emoji raises.
2. Demonstrates the fix: errors="replace" no longer raises.
3. Faithful test: execs the REAL LogInterceptor class from ComfyUI-Turbo
   app/logger.py and writes the exact upscaler line through it.
4. Re-scans all custom nodes for remaining non-cp950 print() lines.
"""
import io
import sys
import pathlib
import codecs
import datetime
import threading

# The console here is cp950; report results without crashing on them.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ok = True

# --- 1. old (strict) behavior ---------------------------------------------
s = "x" * 170 + "\u2705"
buf = io.BytesIO()
w = io.TextIOWrapper(buf, encoding="cp950")
try:
    w.write(s)
    w.flush()
    print("OLD(strict): no error -- unexpected!")
    ok = False
except UnicodeEncodeError as e:
    print(f"OLD(strict): raises UnicodeEncodeError: {e}")

# --- 2. new (errors=replace) behavior -------------------------------------
buf2 = io.BytesIO()
w2 = io.TextIOWrapper(buf2, encoding="cp950", errors="replace")
w2.write(s)
w2.flush()
out = buf2.getvalue()
repl = out[170:171].hex()
print(f"NEW(replace): wrote {len(out)} bytes; emoji byte -> 0x{repl} "
      f"({'0x3f = replaced, no raise' if repl == '3f' else 'UNEXPECTED'})")
if repl != "3f":
    ok = False

# --- 3. faithful test of the actual LogInterceptor class ------------------
src = pathlib.Path(
    r"E:\Comfy\ComfyUI\ComfyUI-Turbo\app\logger.py"
).read_text(encoding="utf-8")
start = src.index("class LogInterceptor")
rest = src[start:]
lines = rest.splitlines(True)
cut = len(lines[0]) if lines else 0
for ln in lines[1:]:
    if ln.startswith("def ") or ln.startswith("class "):
        break
    cut += len(ln)
cls_src = rest[:cut]

from datetime import datetime as _dt
ns = {"io": io, "threading": threading, "datetime": _dt, "logs": []}
exec("import io, threading\nfrom datetime import datetime\n" + cls_src, ns)

fake_stream = io.TextIOWrapper(io.BytesIO(), encoding="cp950")
inter = ns["LogInterceptor"](fake_stream)
try:
    inter.write("[MinimaxH3-3D] \u2705 Model offloaded to CPU. VRAM released.\n")
    inter.flush()
    print("LogInterceptor(real code): wrote the emoji line, no raise -> fix verified")
except UnicodeEncodeError as e:
    print(f"LogInterceptor(real code): STILL RAISES -> {e}")
    ok = False

# --- 4. scan custom nodes for remaining non-cp950 print lines ------------
cp = codecs.getencoder("cp950")
root = pathlib.Path(r"E:\Comfy\ComfyUI\ComfyUI\custom_nodes")
found = 0
for p in root.rglob("*.py"):
    try:
        t = p.read_text(encoding="utf-8")
    except Exception as e:
        print(f"READ-FAIL {p}: {e}")
        continue
    for i, line in enumerate(t.splitlines(), 1):
        if "print(" in line:
            try:
                cp(line)
            except UnicodeEncodeError:
                print(f"REMAINING: {p.parent.name}:{i}: {line.strip()}")
                found += 1
print(f"scan complete: remaining non-cp950 print() lines = {found}")
if found:
    ok = False

print("RESULT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
