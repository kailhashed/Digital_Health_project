# Emotion Dataset Organizer PowerShell Script
# Processes RAVDESS, CREMA-D, and TESS datasets and creates a unified structure

param(
    [string]$SourcePath = ".\Dataset",
    [string]$TargetPath = "."
)

# Emotion mappings
$RAVDESS_EMOTIONS = @{
    '01' = 'neutral'
    '02' = 'calm'
    '03' = 'happy'
    '04' = 'sad'
    '05' = 'angry'
    '06' = 'fearful'
    '07' = 'disgust'
    '08' = 'surprised'
}

$CREMA_EMOTIONS = @{
    'ANG' = 'angry'
    'DIS' = 'disgust'
    'FEA' = 'fearful'
    'HAP' = 'happy'
    'NEU' = 'neutral'
    'SAD' = 'sad'
}

$TESS_EMOTIONS = @{
    'angry' = 'angry'
    'disgust' = 'disgust'
    'fear' = 'fearful'
    'happy' = 'happy'
    'neutral' = 'neutral'
    'sad' = 'sad'
    'pleasant_surprise' = 'surprised'
}

function Create-EmotionFolders {
    param([string]$BasePath)
    
    $emotions = @('angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised')
    
    foreach ($emotion in $emotions) {
        $emotionPath = Join-Path $BasePath "organized_by_emotion\$emotion"
        if (!(Test-Path $emotionPath)) {
            New-Item -ItemType Directory -Path $emotionPath -Force | Out-Null
            Write-Host "Created folder: $emotionPath"
        }
    }
}

function Process-RAVDESSFiles {
    param([string]$SourcePath, [string]$TargetPath)
    
    Write-Host "`nProcessing RAVDESS files..."
    
    $ravdessFiles = Get-ChildItem -Path "$SourcePath\RAVDESS\Actor_*" -Filter "*.wav" -Recurse
    
    foreach ($file in $ravdessFiles) {
        $filename = $file.Name
        
        # Extract emotion from filename: 03-01-01-01-01-01-01.wav
        # The 3rd part (index 2) contains the emotion code
        $parts = $filename.Split('-')
        if ($parts.Length -ge 3) {
            $emotionCode = $parts[2]
            if ($RAVDESS_EMOTIONS.ContainsKey($emotionCode)) {
                $emotion = $RAVDESS_EMOTIONS[$emotionCode]
                
                # Create new filename with source info
                $newFilename = "RAVDESS_$filename"
                $targetFile = Join-Path $TargetPath "organized_by_emotion\$emotion\$newFilename"
                
                # Copy file
                Copy-Item $file.FullName $targetFile
                Write-Host "Copied: $filename -> $emotion\$newFilename"
            }
        }
    }
    
    Write-Host "Processed $($ravdessFiles.Count) RAVDESS files"
}

function Process-CREMAFiles {
    param([string]$SourcePath, [string]$TargetPath)
    
    Write-Host "`nProcessing CREMA-D files..."
    
    $cremaFiles = Get-ChildItem -Path "$SourcePath\CREMA-D\AudioWAV" -Filter "*.wav"
    
    foreach ($file in $cremaFiles) {
        $filename = $file.Name
        
        # Extract emotion from filename: 1001_DFA_ANG_XX.wav
        # The 3rd part (index 2) contains the emotion code
        $parts = $filename.Split('_')
        if ($parts.Length -ge 3) {
            $emotionCode = $parts[2]
            if ($CREMA_EMOTIONS.ContainsKey($emotionCode)) {
                $emotion = $CREMA_EMOTIONS[$emotionCode]
                
                # Create new filename with source info
                $newFilename = "CREMA_$filename"
                $targetFile = Join-Path $TargetPath "organized_by_emotion\$emotion\$newFilename"
                
                # Copy file
                Copy-Item $file.FullName $targetFile
                Write-Host "Copied: $filename -> $emotion\$newFilename"
            }
        }
    }
    
    Write-Host "Processed $($cremaFiles.Count) CREMA-D files"
}

function Process-TESSFiles {
    param([string]$SourcePath, [string]$TargetPath)
    
    Write-Host "`nProcessing TESS files..."
    
    $tessBase = "$SourcePath\TESS\TESS Toronto emotional speech set data\TESS Toronto emotional speech set data"
    
    # Process each emotion folder
    $emotionFolders = Get-ChildItem -Path $tessBase -Directory
    
    foreach ($folder in $emotionFolders) {
        $folderName = $folder.Name
        
        # Map folder name to emotion
        $emotion = $null
        foreach ($key in $TESS_EMOTIONS.Keys) {
            if ($folderName.ToLower().Contains($key.ToLower())) {
                $emotion = $TESS_EMOTIONS[$key]
                break
            }
        }
        
        if ($emotion) {
            # Find all wav files in this folder
            $wavFiles = Get-ChildItem -Path $folder.FullName -Filter "*.wav"
            
            foreach ($file in $wavFiles) {
                $filename = $file.Name
                
                # Create new filename with source info
                $newFilename = "TESS_$filename"
                $targetFile = Join-Path $TargetPath "organized_by_emotion\$emotion\$newFilename"
                
                # Copy file
                Copy-Item $file.FullName $targetFile
                Write-Host "Copied: $filename -> $emotion\$newFilename"
            }
            
            Write-Host "Processed $($wavFiles.Count) files from $folderName -> $emotion"
        }
    }
}

function Check-VideoFiles {
    param([string]$SourcePath)
    
    Write-Host "`nChecking for video files..."
    
    $videoExtensions = @('*.mp4', '*.avi', '*.mov', '*.mkv', '*.flv', '*.wmv')
    $videoFiles = @()
    
    foreach ($ext in $videoExtensions) {
        $videoFiles += Get-ChildItem -Path $SourcePath -Filter $ext -Recurse -ErrorAction SilentlyContinue
    }
    
    if ($videoFiles.Count -gt 0) {
        Write-Host "Found $($videoFiles.Count) video files that need conversion:"
        foreach ($videoFile in $videoFiles) {
            Write-Host "  - $($videoFile.FullName)"
        }
        return $true
    } else {
        Write-Host "No video files found. All files are already in audio format."
        return $false
    }
}

# Main execution
Write-Host "Emotion Dataset Organizer"
Write-Host "=" * 50
Write-Host "Source dataset path: $SourcePath"
Write-Host "Target path: $TargetPath"

# Check if dataset folder exists
if (!(Test-Path $SourcePath)) {
    Write-Host "Error: Dataset folder not found at $SourcePath"
    exit 1
}

# Create emotion-based folder structure
Create-EmotionFolders -BasePath $TargetPath

# Process each dataset
Process-RAVDESSFiles -SourcePath $SourcePath -TargetPath $TargetPath
Process-CREMAFiles -SourcePath $SourcePath -TargetPath $TargetPath
Process-TESSFiles -SourcePath $SourcePath -TargetPath $TargetPath

# Check for video files
$hasVideoFiles = Check-VideoFiles -SourcePath $SourcePath

Write-Host "`n" + "=" * 50
Write-Host "Dataset organization complete!"

# Print summary
$organizedPath = Join-Path $TargetPath "organized_by_emotion"
if (Test-Path $organizedPath) {
    Write-Host "`nOrganized dataset created at: $organizedPath"
    
    $emotionFolders = Get-ChildItem -Path $organizedPath -Directory
    foreach ($folder in $emotionFolders) {
        $fileCount = (Get-ChildItem -Path $folder.FullName -Filter "*.wav").Count
        Write-Host "  $($folder.Name): $fileCount files"
    }
}

if ($hasVideoFiles) {
    Write-Host "`nNote: Video files were found but not converted."
    Write-Host "All current files are already in audio format (.wav)"
}
