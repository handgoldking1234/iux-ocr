# IUX 엔진 모델

인터넷·GPU 없이 CPU(ONNX Runtime)로 동작하는 내장 OCR 모델입니다.

| 파일 | 역할 | 출처 | 라이선스 |
|---|---|---|---|
| `iux_det.onnx` | 글줄 찾기 | PaddleOCR PP-OCRv5 mobile det (ONNX 변환: GreatV/oar-ocr) | Apache-2.0 |
| `iux_rec_ko.onnx` | 한글 읽기(인쇄체 위주) | PaddleOCR korean_PP-OCRv5_mobile_rec | Apache-2.0 |
| `iux_rec_hw.onnx` | 한글 손글씨 읽기 | VerifiedIdiot/korean-handwriting-rec v1.0.0 (PP-OCRv5 한국어 모델을 AI Hub 손글씨·공문서 460만 단어로 추가 학습) | Apache-2.0 |
| `iux_rec_zh.onnx` | 한자 읽기 (한국 한자 KS X 1001로 출력 제한) | PaddleOCR PP-OCRv6 small rec (ONNX 변환: GreatV/oar-ocr) | Apache-2.0 |
| `iux_rec_zh_dict.txt` | 한자 모델 글자표 | PaddleOCR | Apache-2.0 |
| `iux_rec_ko_dict.txt` | 글자표(한글 11,172자 포함) | PaddleOCR | Apache-2.0 |

손글씨 모델은 AI Hub(www.aihub.or.kr) "대용량 손글씨 OCR 데이터", "다양한 형태의 한글 문자 OCR" 데이터로 학습된 공개 모델입니다.
IUX 엔진은 한자 모델이 확신하는 한자를 글자 위치로 맞춰 끼워 넣고, 줄마다 두 한글 모델 중 더 확신하는 결과를 고르고, 글줄 이미지의 빈칸으로 띄어쓰기를 되살립니다(`app/iux_engine.py`).
