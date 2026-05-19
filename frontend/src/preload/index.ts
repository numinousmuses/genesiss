import { contextBridge } from "electron";

const bridgeArg = process.argv.find((a) => a.startsWith("--bridge-url="));
const bridgeUrl = bridgeArg?.split("=")[1] ?? "http://127.0.0.1:8765";

contextBridge.exposeInMainWorld("genesiss", {
  bridgeUrl,
});

declare global {
  interface Window {
    genesiss: { bridgeUrl: string };
  }
}
