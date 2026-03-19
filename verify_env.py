import sys, importlib.metadata, torch

results = []

def check(name, fn):
    try:
        results.append((name, "OK", fn()))
    except Exception as e:
        results.append((name, "FAIL", str(e)))

def pkg_ver(pkg_name):
    """Safely get installed package version via importlib.metadata."""
    return importlib.metadata.version(pkg_name)

check("Python >= 3.10", lambda: sys.version.split()[0])
check("torch",         lambda: pkg_ver("torch"))
check("transformers",  lambda: pkg_ver("transformers"))
check("accelerate",    lambda: pkg_ver("accelerate"))
check("peft",          lambda: pkg_ver("peft"))
check("datasets",      lambda: pkg_ver("datasets"))
check("gradio",        lambda: pkg_ver("gradio"))
check("langdetect",    lambda: pkg_ver("langdetect"))
check("sentencepiece", lambda: pkg_ver("sentencepiece"))

if torch.cuda.is_available():
    check("CUDA GPU", lambda: torch.cuda.get_device_name(0))
elif torch.backends.mps.is_available():
    check("Apple MPS", lambda: "Metal GPU ✓")
else:
    check("Device", lambda: "CPU only")

try:
    check("mlx", lambda: pkg_ver("mlx"))
except: pass

print("\n=== Environment Check ===")
for name, status, val in results:
    print(f"  {'✓' if status=='OK' else '✗'} {name:<20} {val}")
fails = sum(1 for _, s, _ in results if s == "FAIL")
print(f"\n{'All OK!' if not fails else f'{fails} lỗi cần sửa'}")
