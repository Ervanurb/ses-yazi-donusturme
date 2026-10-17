# Windows'un yerleşik Türkçe sesiyle (Tolga) deneme kaydı üretir.
# Kullanım: powershell -File scripts\tts_ornek.ps1 -Cikti samples\tts.wav
param(
    [string]$Cikti = "samples\tts.wav",
    [string]$Metin = "Merhaba, bu bir ses işleme denemesidir. Bugün sesi yazıya dönüştüren küçük bir araç yapıyoruz. Önce sesi temizliyoruz, sonra konuşma olan bölümleri buluyoruz. En sonunda Whisper modeli bu konuşmayı metne çeviriyor."
)

Add-Type -AssemblyName System.Runtime.WindowsRuntime
$null = [Windows.Media.SpeechSynthesis.SpeechSynthesizer, Windows.Media.SpeechSynthesis, ContentType = WindowsRuntime]
$null = [Windows.Storage.Streams.DataReader, Windows.Storage.Streams, ContentType = WindowsRuntime]

# WinRT'nin async metotlarını PowerShell'de beklemek için yardımcı
$asTask = ([System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object {
    $_.Name -eq 'AsTask' -and $_.GetParameters().Count -eq 1 -and $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1'
})[0]
function Await($op, [Type]$type) {
    $task = $asTask.MakeGenericMethod($type).Invoke($null, @($op))
    $task.Wait() | Out-Null
    $task.Result
}

$synth = New-Object Windows.Media.SpeechSynthesis.SpeechSynthesizer
$voice = [Windows.Media.SpeechSynthesis.SpeechSynthesizer]::AllVoices | Where-Object { $_.Language -eq 'tr-TR' } | Select-Object -First 1
if (-not $voice) { throw "Türkçe ses bulunamadı" }
$synth.Voice = $voice

$stream = Await ($synth.SynthesizeTextToStreamAsync($Metin)) ([Windows.Media.SpeechSynthesis.SpeechSynthesisStream])
$reader = New-Object Windows.Storage.Streams.DataReader($stream.GetInputStreamAt(0))
$size = [uint32]$stream.Size
$null = Await ($reader.LoadAsync($size)) ([uint32])
$bytes = New-Object byte[] $size
$reader.ReadBytes($bytes)

$dir = Split-Path -Parent $Cikti
if ($dir) { New-Item -ItemType Directory -Force $dir | Out-Null }
[System.IO.File]::WriteAllBytes((Join-Path (Get-Location) $Cikti), $bytes)
Write-Output "Kaydedildi: $Cikti ($($voice.DisplayName))"
