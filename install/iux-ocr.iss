; IUX OCR 윈도우 설치 파일 (Inno Setup 6)
#ifndef AppVersion
  #define AppVersion "0.3.0"
#endif

[Setup]
AppId=IUX-OCR-Local
AppName=IUX OCR
AppVersion={#AppVersion}
AppVerName=IUX OCR {#AppVersion}
AppPublisher=IUX
DefaultDirName={autopf}\IUX OCR
DefaultGroupName=IUX OCR
DisableProgramGroupPage=yes
OutputDir=..\dist
OutputBaseFilename=IUX-OCR-Setup
Compression=lzma2/max
SolidCompression=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
WizardStyle=modern
UninstallDisplayIcon={app}\IUX OCR.exe

[Languages]
#ifdef KOREAN
Name: "korean"; MessagesFile: "compiler:Languages\Korean.isl"
#else
Name: "english"; MessagesFile: "compiler:Default.isl"
#endif

[Tasks]
Name: "desktopicon"; Description: "바탕화면에 바로가기 만들기"; Flags: checkedonce

[Files]
Source: "..\dist\IUX OCR\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\IUX OCR"; Filename: "{app}\IUX OCR.exe"
Name: "{group}\IUX OCR 제거"; Filename: "{uninstallexe}"
Name: "{autodesktop}\IUX OCR"; Filename: "{app}\IUX OCR.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\IUX OCR.exe"; Description: "IUX OCR 바로 실행"; Flags: nowait postinstall skipifsilent
