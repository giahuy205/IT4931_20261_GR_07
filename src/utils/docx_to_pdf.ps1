$docxPath = "D:\Hust\BigData\[NHÓM 7] BÁO CÁO TIẾN ĐỘ DỰ ÁN BIG DATA.docx"
$pdfPath = "D:\Hust\BigData\[NHÓM 7] BÁO CÁO TIẾN ĐỘ DỰ ÁN BIG DATA.pdf"

try {
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $doc = $word.Documents.Open($docxPath)
    $doc.SaveAs([ref]$pdfPath, [ref]17) # 17 = wdFormatPDF
    $doc.Close()
    $word.Quit()
    Write-Output "PDF generated successfully!"
} catch {
    Write-Output "Error: $($_.Exception.Message)"
}
