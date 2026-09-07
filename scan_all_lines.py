"""List ALL lines containing non-cp950 chars in the H3 upscaler package files."""
import sys
import pathlib
import codecs

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

cp = codecs.getencoder("cp950")
root = pathlib.Path(r"E:\Comfy\ComfyUI\ComfyUI\custom_nodes\Comfyui_Minimax_h3_latent_Upscaler")
for p in sorted(root.rglob("*.py")):
    try:
        t = p.read_text(encoding="utf-8")
    except Exception as e:
        print(f"READ-FAIL {p}: {e}")
        continue
    bad = []
    for i, line in enumerate(t.splitlines(), 1):
        try:
            cp(line)
        except UnicodeEncodeError:
            bad.append(i)
    if bad:
        print(f"== {p.name}: lines {bad}")
