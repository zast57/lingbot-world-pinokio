module.exports = {
  version: "5.0",
  menu: async (kernel, info) => {
    const installing = info.running("install.js")
    const updating = info.running("update.js")
    const resetting = info.running("reset.js")
    const downloading = info.running("download_model.js")
    const running = info.running("start.js")

    const installed = info.exists("app/env")
    const modelNF4 = info.exists("app/lingbot-world-base-cam-nf4")
    const modelOriginal = info.exists("app/lingbot-world-base-cam")
    const modelsReady = modelNF4 && modelOriginal

    if (installing) {
      return [{
        default: true,
        icon: "fa-solid fa-plug",
        text: "Installing",
        href: "install.js",
      }]
    }

    if (updating) {
      return [{
        default: true,
        icon: "fa-solid fa-arrows-rotate",
        text: "Updating",
        href: "update.js",
      }]
    }

    if (resetting) {
      return [{
        default: true,
        icon: "fa-solid fa-broom",
        text: "Resetting",
        href: "reset.js",
      }]
    }

    if (downloading) {
      return [{
        default: true,
        icon: "fa-solid fa-download",
        text: "Downloading Models",
        href: "download_model.js",
      }]
    }

    if (!installed) {
      return [{
        default: true,
        icon: "fa-solid fa-plug",
        text: "Install",
        href: "install.js",
      }]
    }

    if (running) {
      let local = info.local("start.js")
      if (local && local.url) {
        return [{
          default: true,
          icon: "fa-solid fa-desktop",
          text: "Open Web UI",
          href: local.url,
        }, {
          icon: "fa-solid fa-terminal",
          text: "Terminal",
          href: "start.js",
        }]
      } else {
        return [{
          default: true,
          icon: "fa-solid fa-terminal",
          text: "Starting...",
          href: "start.js",
        }]
      }
    }

    let menu = []

    if (modelsReady) {
      menu.push({
        default: true,
        icon: "fa-solid fa-play",
        text: "Start",
        href: "start.js"
      })
    }

    menu.push(
      {
        icon: "fa-solid fa-bug",
        text: "DEBUG (Test Script)",
        href: "debug.js"
      },
      {
        icon: "fa-solid fa-download",
        text: modelsReady ? "Re-download Models" : "Download Models (~28GB)",
        href: "download_model.js"
      },
      {
        icon: "fa-solid fa-arrows-rotate",
        text: "Update",
        href: "update.js"
      },
      {
        icon: "fa-solid fa-broom",
        text: "Reset",
        href: "reset.js"
      },
      {
        icon: "fa-solid fa-trash",
        text: "Uninstall",
        href: "uninstall.js"
      }
    )

    return menu
  }
}
