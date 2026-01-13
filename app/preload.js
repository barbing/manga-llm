const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("mangaApp", {
  openFolder: () => ipcRenderer.invoke("dialog:openFolder"),
  openFiles: () => ipcRenderer.invoke("dialog:openFiles"),
});
