module.exports = {
  run: [
    {
      method: "fs.rm",
      params: {
        path: "app/env"
      }
    },
    {
      method: "notify",
      params: {
        html: "🔄 Reset complete. Click <b>Install</b> to reinstall."
      }
    }
  ]
}
