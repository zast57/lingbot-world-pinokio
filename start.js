module.exports = {
  daemon: true,
  run: [
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        message: "uv pip install gradio",
      }
    },
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        message: "python app_gradio.py",
        on: [{
          event: "/http:\\/\\/[0-9.:]+/",
          done: true
        }]
      }
    },
    {
      method: "local.set",
      params: {
        url: "{{input.event[0]}}"
      }
    }
  ]
}
