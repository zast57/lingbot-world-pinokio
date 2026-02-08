module.exports = {
    run: [
        {
            method: "shell.run",
            params: {
                path: "app",
                message: "rm -rf env",
            }
        },
        {
            method: "notify",
            params: {
                html: "Uninstalled Environment",
                href: "start.js"
            }
        }
    ]
}
