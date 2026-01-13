const { createApp } = Vue;

createApp({
  data() {
    return {
      inputFolder: "",
      inputFiles: [],
      outputFolder: "",
      targetLang: "zh-CN",
      model: "huihui_ai/qwen3-abliterated:14b",
      style: "modern_casual",
      glossaryPath: "",
      progress: 0,
      status: "Waiting for input",
      eta: "--",
    };
  },
  methods: {
    async pickFolder() {
      const folder = await window.mangaApp.openFolder();
      if (folder) {
        this.inputFolder = folder;
        this.inputFiles = [];
        this.status = "Ready to start";
      }
    },
    async pickFiles() {
      const files = await window.mangaApp.openFiles();
      if (files.length) {
        this.inputFiles = files;
        this.inputFolder = "";
        this.status = "Ready to start";
      }
    },
    async pickOutput() {
      const folder = await window.mangaApp.openFolder();
      if (folder) {
        this.outputFolder = folder;
      }
    },
    start() {
      if (!this.inputFolder && this.inputFiles.length === 0) {
        this.status = "Please select an input folder or images";
        return;
      }
      if (!this.outputFolder) {
        this.status = "Please choose an output folder";
        return;
      }
      this.status = "Queued: preparing pipeline";
      this.progress = 5;
      this.eta = "estimating";
      setTimeout(() => {
        this.progress = 40;
        this.status = "OCR + bubble detection";
      }, 600);
      setTimeout(() => {
        this.progress = 70;
        this.status = "Translating with Ollama";
      }, 1200);
      setTimeout(() => {
        this.progress = 100;
        this.status = "Rendering complete";
        this.eta = "00:00";
      }, 1800);
    },
  },
}).mount("#app");
