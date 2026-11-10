# 설치 파일 검증: 실제 설치 → 설치된 exe 실행 → 주요 기능 점검 → 제거
$ErrorActionPreference = "Stop"
$results = @()
function Check($name, [scriptblock]$body) {
  try { $msg = & $body; $script:results += [pscustomobject]@{ 항목 = $name; 결과 = "통과"; 내용 = "$msg" } }
  catch { $script:results += [pscustomobject]@{ 항목 = $name; 결과 = "실패"; 내용 = "$($_.Exception.Message)" } }
}

# 한글 사용자 폴더 환경 재현
$ko = Join-Path $env:RUNNER_TEMP "사용자한글"
New-Item -ItemType Directory -Force -Path $ko | Out-Null
$env:TEMP = $ko; $env:TMP = $ko
$dir = Join-Path $env:RUNNER_TEMP "설치폴더\IUX OCR"

Check "설치 (무인 설치)" {
  $p = Start-Process "dist\IUX-OCR-Setup.exe" -ArgumentList "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART", "/CURRENTUSER", "/DIR=`"$dir`"" -Wait -PassThru
  if ($p.ExitCode -ne 0) { throw "설치 종료 코드 $($p.ExitCode)" }
  if (-not (Test-Path "$dir\IUX OCR.exe")) { throw "exe 없음" }
  "설치 위치 $dir"
}

$base = "http://127.0.0.1:8765"
$proc = $null
Check "설치된 exe 실행" {
  $script:proc = Start-Process "$dir\IUX OCR.exe" -ArgumentList "--no-browser" -PassThru
  $sw = [Diagnostics.Stopwatch]::StartNew()
  for ($i = 0; $i -lt 60; $i++) { Start-Sleep 1; try { $null = Invoke-RestMethod "$base/api/engines"; break } catch {} }
  if ($i -ge 60) { throw "60초 안에 서버가 뜨지 않음" }
  "기동 $([math]::Round($sw.Elapsed.TotalSeconds,1))초"
}
Check "엔진 상태 (Tesseract 한국어)" {
  $e = Invoke-RestMethod "$base/api/engines"
  if (-not $e.tesseract.has_kor) { throw "한국어 없음: $($e.tesseract.reason)" }
  "Tesseract $($e.tesseract.version) / 언어 $($e.tesseract.langs -join ',')"
}
Check "화면(index) 열림" {
  $h = Invoke-WebRequest "$base/" -UseBasicParsing
  if ($h.Content -notmatch "IUX OCR") { throw "화면 내용 이상" }
  "$([math]::Round($h.RawContentLength/1KB))KB"
}

$jpg = Join-Path $ko "공문_샘플.jpg"; Copy-Item (Get-ChildItem samples\*.jpg | Select-Object -First 1).FullName $jpg
$pdf = Join-Path $ko "공문_샘플.pdf"; Copy-Item (Get-ChildItem samples\*.pdf | Select-Object -First 1).FullName $pdf
$docs = @{}
Check "이미지 업로드 (한글 파일명)" { $d = Invoke-RestMethod "$base/api/upload" -Method Post -Form @{ files = Get-Item $jpg }; $script:docs.jpg = $d[0].id; "$($d[0].name)" }
Check "PDF 업로드" { $d = Invoke-RestMethod "$base/api/upload" -Method Post -Form @{ files = Get-Item $pdf }; $script:docs.pdf = $d[0].id; "$($d[0].name), $($d[0].pages.Count)쪽" }

foreach ($k in @("jpg", "pdf")) {
  Check "OCR 인식 ($k)" {
    $sw = [Diagnostics.Stopwatch]::StartNew()
    $r = Invoke-RestMethod "$base/api/docs/$($docs[$k])/ocr" -Method Post -ContentType "application/json; charset=utf-8" -Body ([Text.Encoding]::UTF8.GetBytes('{"page":0,"engine":"tesseract","lang":"kor+eng","psm":4}'))
    if ($r.result.text -notmatch "서울") { throw "인식 결과 이상: $($r.result.text.Substring(0, [Math]::Min(40, $r.result.text.Length)))" }
    "신뢰도 $($r.result.mean_conf), $([math]::Round($sw.Elapsed.TotalSeconds,1))초, $($r.result.lines.Count)줄"
  }
}
Check "IUX 엔진 (내장 딥러닝, 오프라인)" {
  $e = Invoke-RestMethod "$base/api/engines"
  if (-not $e.iux.available) { throw "IUX 엔진 사용 불가: $($e.iux.reason)" }
  $sw = [Diagnostics.Stopwatch]::StartNew()
  $r = Invoke-RestMethod "$base/api/docs/$($docs.jpg)/ocr" -Method Post -ContentType "application/json; charset=utf-8" -Body ([Text.Encoding]::UTF8.GetBytes('{"page":0,"engine":"iux","lang":"kor+eng","psm":4}'))
  if ($r.result.text -notmatch "서울") { throw "IUX 인식 결과 이상: $($r.result.text.Substring(0, [Math]::Min(40, $r.result.text.Length)))" }
  "신뢰도 $($r.result.mean_conf), $([math]::Round($sw.Elapsed.TotalSeconds,1))초, $($r.result.lines.Count)줄"
}
Check "항목 자동 채움 (기록물 메타데이터)" {
  $f = Invoke-RestMethod "$base/api/docs/$($docs.jpg)/autofill" -Method Post -ContentType "application/json" -Body '{"template_id":"records-meta"}'
  $filled = @($f.values | Where-Object { $_.value }).Count
  $t = ($f.values | Where-Object { $_.name -eq "제목" }).value
  if (-not $t) { throw "제목을 못 찾음" }
  "$filled/$($f.values.Count)개 채움, 제목=$t"
}
Check "개인정보 마스킹" {
  $body = [Text.Encoding]::UTF8.GetBytes('{"mode":"mask","text":"연락처 010-1234-5678, 주민 900101-1234567"}')
  $r = Invoke-RestMethod "$base/api/refine" -Method Post -ContentType "application/json; charset=utf-8" -Body $body
  if ($r.text -match "1234-5678|1234567") { throw "마스킹 안 됨: $($r.text)" }
  $r.text
}
$out = Join-Path $ko "내보내기"; New-Item -ItemType Directory -Force -Path $out | Out-Null
foreach ($fmt in @("txt", "md", "docx", "xlsx", "json", "pdf")) {
  Check "내보내기 $fmt" {
    $file = Join-Path $out "result.$fmt"
    Invoke-WebRequest "$base/api/docs/$($docs.jpg)/export" -Method Post -ContentType "application/json" -Body "{`"format`":`"$fmt`",`"template_id`":`"records-meta`"}" -OutFile $file -UseBasicParsing
    $len = (Get-Item $file).Length
    if ($len -lt 200) { throw "파일이 너무 작음 ($len B)" }
    "$([math]::Round($len/1KB,1))KB"
  }
}
Check "작업 폴더 위치" {
  $ws = Join-Path $env:LOCALAPPDATA "IUX OCR\workspace"
  if (-not (Test-Path "$ws\docs")) { throw "작업 폴더 없음: $ws" }
  $ws
}
Check "프로그램 종료" { Stop-Process -Id $proc.Id -Force; "종료" }
Check "제거 (언인스톨)" {
  $u = Get-ChildItem "$dir\unins*.exe" | Select-Object -First 1
  $p = Start-Process $u.FullName -ArgumentList "/VERYSILENT", "/SUPPRESSMSGBOXES" -Wait -PassThru
  Start-Sleep 3
  if (Test-Path "$dir\IUX OCR.exe") { throw "exe가 남아 있음" }
  "제거 완료"
}

$results | Format-Table -AutoSize -Wrap | Out-String -Width 220 | Write-Host
$results | ConvertTo-Json | Set-Content -Encoding utf8 dist\verify-report.json
$fail = @($results | Where-Object { $_.결과 -eq "실패" }).Count
"총 $($results.Count)개 중 실패 $fail 개"
if ($fail -gt 0) { exit 1 }
