module.exports = {
  run: [
    {
      method: "shell.run",
      params: {
        path: "app",
        message: "git pull"
      }
    },
    {
      method: "notify",
      params: {
        html: "✅ Updated!"
      }
    }
  ]
}
