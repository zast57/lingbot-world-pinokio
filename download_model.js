module.exports = {
  run: [
    // 1. Download NF4 quantized model (~19GB)
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        message: [
          "python -c \"from huggingface_hub import snapshot_download; snapshot_download(repo_id='cahlen/lingbot-world-base-cam-nf4', local_dir='./lingbot-world-base-cam-nf4')\"",
          "python -c \"from huggingface_hub import snapshot_download; snapshot_download(repo_id='robbyant/lingbot-world-base-cam', local_dir='./lingbot-world-base-cam')\""
        ]
      }
    },
    {
      method: "notify",
      params: {
        html: "✅ Models downloaded!<br><br>Click <b>Start</b> to generate videos."
      }
    }
  ]
}
