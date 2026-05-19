import { app, BrowserWindow, shell } from "electron";
import { join } from "node:path";

const isDev = !app.isPackaged;
const bridgeUrl = process.env.GENESISS_BRIDGE_URL ?? "http://127.0.0.1:8765";

function createWindow() {
  const win = new BrowserWindow({
    width: 1400,
    height: 900,
    title: "Genesiss",
    backgroundColor: "#0b0b0f",
    webPreferences: {
      preload: join(__dirname, "../preload/index.js"),
      contextIsolation: true,
      nodeIntegration: false,
      additionalArguments: [`--bridge-url=${bridgeUrl}`],
    },
  });

  win.webContents.setWindowOpenHandler(({ url }) => {
    shell.openExternal(url);
    return { action: "deny" };
  });

  if (isDev && process.env.ELECTRON_RENDERER_URL) {
    win.loadURL(process.env.ELECTRON_RENDERER_URL);
    win.webContents.openDevTools({ mode: "detach" });
  } else {
    win.loadFile(join(__dirname, "../renderer/index.html"));
  }
}

app.whenReady().then(createWindow);

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit();
});

app.on("activate", () => {
  if (BrowserWindow.getAllWindows().length === 0) createWindow();
});
