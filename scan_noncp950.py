"""Extended scan: find print/raise/logger lines that cp950 cannot encode."""
import sys
import pathlib
import codecs

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

cp = codecs.getencoder("cp950")
root = pathlib.Path(r"E:\Comfy\ComfyUI\ComfyUI\custom_nodes")
names = ("Comfyui_Minimax_h3_latent_Upscaler", "ComfyUI-MiniMax-H3-Turbo",
         "ComfyUI-H3-Motion-Context", "ComfyUI-MiniMax-H3-Guide",
         "minimax-h3-audio-T8", "Comfyui_Minimax_h3")
found = 0
for p in root.rglob("*.py"):
    try:
        t = p.read_text(encoding="utf-8")
    except Exception as e:
        print(f"READ-FAIL {p}: {e}")
        continue
    top = p.parts[-2] if len(p.parts) > 2 else p.parent.name
    for i, line in enumerate(t.splitlines(), 1):
        if not any(k in line for k in ("print(", "raise ", "logger.", "logging.")):
            continue
        try:
            cp(line)
        except UnicodeEncodeError as e:
            print(f"{top}:{i}: {line.strip()}")
            found += 1
print(f"total non-cp950 log/raise lines: {found}")
