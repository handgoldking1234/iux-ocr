"""IUX OCR 실행 진입점 (설치형 exe 빌드용)."""
import multiprocessing

from app.server import main

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
