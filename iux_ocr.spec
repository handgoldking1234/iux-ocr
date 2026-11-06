# PyInstaller 빌드 설정 — pyinstaller iux_ocr.spec
from PyInstaller.utils.hooks import collect_submodules

hidden = collect_submodules("uvicorn") + ["multipart", "python_multipart", "pypdfium2", "reportlab.pdfbase.cidfonts"]

a = Analysis(
    ["iux_ocr.py"],
    pathex=[],
    datas=[("app/static", "app/static")],
    hiddenimports=hidden,
    excludes=["paddle", "paddleocr", "matplotlib", "IPython", "pytest", "scipy", "pandas", "torch", "tensorflow", "sklearn", "notebook", "jupyter", "playwright", "PyQt5", "PyQt6", "PySide6"],
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="IUX OCR",
    console=True,
    icon="install/iux-ocr.ico" if __import__("os").path.exists("install/iux-ocr.ico") else None,
)
coll = COLLECT(exe, a.binaries, a.datas, name="IUX OCR")
