module.exports = {
    daemon: true,
    run: [
        {
            method: "shell.run",
            params: {
                message: "echo HELLO WORLD DEBUG"
            }
        }
    ]
}
