# 启动数字卫兵程序，过滤剪贴板错误消息
# 用法: .\run_sentinel.ps1

Write-Host "🚀 启动数字卫兵..." -ForegroundColor Green

# 方法：使用 Start-Process 并捕获输出，过滤错误消息
$process = Start-Process -FilePath "cargo" -ArgumentList "run", "--bin", "sentinel" -NoNewWindow -PassThru -RedirectStandardError ".\stderr.log"

# 实时显示 stderr 但过滤剪贴板错误
$job = Start-Job -ScriptBlock {
    Get-Content ".\stderr.log" -Wait -Tail 0 | Where-Object { 
        $_ -notmatch "Failed to open clipboard" 
    } | ForEach-Object { 
        Write-Error $_
    }
}

# 等待进程结束
$process.WaitForExit()

# 停止监控任务
Stop-Job $job
Remove-Job $job

# 清理临时文件
Remove-Item ".\stderr.log" -ErrorAction SilentlyContinue

Write-Host "`n程序已退出" -ForegroundColor Yellow
