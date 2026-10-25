// frontend/src/video_io.ts
import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";
import { ComfyWidgets } from "../../../scripts/widgets.js";
function transient(widget) {
  widget.serialize = false;
  widget.options ??= {};
  widget.options.serialize = false;
  return widget;
}
function infoText(info) {
  return `${info.codec} / ${info.pixel_format}
${info.width} \xD7 ${info.height}; ${info.fps.toFixed(6)} fps
\u041A\u0430\u0434\u0440\u043E\u0432: ${info.total_frames}; \u043D\u043E\u043C\u0435\u0440\u0430 0\u2013${info.total_frames - 1}` + (info.selected_frames ? `
\u0412\u044B\u0431\u0440\u0430\u043D\u043E: ${info.start_frame}\u2013${info.end_frame}, \u043A\u0430\u0434\u0440\u043E\u0432: ${info.selected_frames}` : "");
}
var loaders = /* @__PURE__ */ new WeakMap();
var players = /* @__PURE__ */ new WeakMap();
function ensureLoader(node) {
  let state = loaders.get(node);
  if (!state) {
    const info = transient(ComfyWidgets.STRING(node, "video_info", ["STRING", { multiline: true }], app).widget);
    if (info.inputEl) info.inputEl.readOnly = true;
    info.value = "\u0423\u043A\u0430\u0436\u0438\u0442\u0435 \u043F\u0443\u0442\u044C \u043A \u0432\u0438\u0434\u0435\u043E \u0438\u043B\u0438 \u0432\u044B\u0431\u0435\u0440\u0438\u0442\u0435 \u0444\u0430\u0439\u043B. \u0417\u0430\u0442\u0435\u043C Load and calculate frame.";
    const input = document.createElement("input");
    input.type = "file";
    input.accept = "video/*,.mov,.mxf,.mkv,.avi,.mp4,.m4v,.webm,.mpg,.mpeg,.ts";
    input.style.display = "none";
    document.body.append(input);
    const choose = transient(node.addWidget("button", "\u0412\u044B\u0431\u0440\u0430\u0442\u044C \u0432\u0438\u0434\u0435\u043E\u2026", null, () => input.click()));
    const button = transient(node.addWidget("button", "Load and calculate frame", null, () => {
      void probe();
    }));
    state = { input, info, choose, button, busy: false };
    loaders.set(node, state);
    const current = state;
    const pathWidget = () => node.widgets.find((widget) => widget.name === "video_path");
    const changed = pathWidget()?.callback;
    const path = pathWidget();
    if (path) path.callback = function(...args) {
      current.request?.abort();
      current.info.value = "\u0424\u0430\u0439\u043B \u0438\u0437\u043C\u0435\u043D\u0451\u043D. \u041D\u0430\u0436\u043C\u0438\u0442\u0435 Load and calculate frame.";
      return changed?.apply(this, args);
    };
    async function probe() {
      current.request?.abort();
      const controller = new AbortController();
      current.request = controller;
      const path2 = String(pathWidget()?.value ?? "");
      current.info.value = "\u0427\u0442\u0435\u043D\u0438\u0435 \u0438\u043D\u0444\u043E\u0440\u043C\u0430\u0446\u0438\u0438\u2026 \u0415\u0441\u043B\u0438 \u0432 \u043A\u043E\u043D\u0442\u0435\u0439\u043D\u0435\u0440\u0435 \u043D\u0435\u0442 \u0447\u0438\u0441\u043B\u0430 \u043A\u0430\u0434\u0440\u043E\u0432, \u0432\u044B\u043F\u043E\u043B\u043D\u044F\u0435\u0442\u0441\u044F \u043F\u043E\u0434\u0441\u0447\u0451\u0442 \u0431\u0435\u0437 \u0441\u043E\u0445\u0440\u0430\u043D\u0435\u043D\u0438\u044F \u0438\u0437\u043E\u0431\u0440\u0430\u0436\u0435\u043D\u0438\u0439.";
      try {
        const response = await api.fetchApi(`/alexz/video/probe?path=${encodeURIComponent(path2)}`, { signal: controller.signal });
        const data = await response.json();
        if (!response.ok) throw Error(response.status === 404 ? "\u041F\u0435\u0440\u0435\u0437\u0430\u043F\u0443\u0441\u0442\u0438\u0442\u0435 ComfyUI \u0434\u043B\u044F \u0437\u0430\u0433\u0440\u0443\u0437\u043A\u0438 \u043C\u0430\u0440\u0448\u0440\u0443\u0442\u0430 \u0438\u043D\u0444\u043E\u0440\u043C\u0430\u0446\u0438\u0438 \u043E \u0432\u0438\u0434\u0435\u043E." : data.error ?? `HTTP ${response.status}`);
        if (path2 === String(pathWidget()?.value ?? "")) current.info.value = infoText(data);
      } catch (error) {
        if (!controller.signal.aborted) current.info.value = String(error);
      }
      node.graph?.setDirtyCanvas(true, true);
    }
    input.onchange = async () => {
      if (!input.files?.[0] || current.busy) return;
      current.busy = true;
      current.info.value = "\u0417\u0430\u0433\u0440\u0443\u0437\u043A\u0430 \u0444\u0430\u0439\u043B\u0430 \u0432 ComfyUI/input\u2026";
      try {
        const file = input.files[0];
        const response = await api.fetchApi(`/alexz/video/upload?filename=${encodeURIComponent(file.name)}`, {
          method: "POST",
          headers: { "Content-Type": "application/octet-stream", "X-Alexz-Video": "1" },
          body: file
        });
        const data = await response.json();
        if (!response.ok || !data.name) throw Error(data.error ?? `\u0417\u0430\u0433\u0440\u0443\u0437\u043A\u0430: HTTP ${response.status}`);
        const widget = pathWidget();
        if (widget) {
          widget.value = [data.subfolder, data.name].filter(Boolean).join("/");
          widget.callback?.(widget.value);
        }
        await probe();
      } catch (error) {
        current.info.value = String(error);
      } finally {
        current.busy = false;
        input.value = "";
      }
    };
  }
  node.widgets = [state.choose, state.button, state.info, ...node.widgets.filter((widget) => ![state.choose, state.button, state.info].includes(widget))];
}
app.registerExtension({
  name: "ALEXZ.Tools.VideoIO",
  beforeRegisterNodeDef(...args) {
    const nodeType = args[0];
    const name = args[1].name;
    if (!["VideoLoadSource", "VideoSaveFinal"].includes(name)) return;
    if (name === "VideoLoadSource") {
      for (const hook of ["onNodeCreated", "onConfigure"]) {
        const previous = nodeType.prototype[hook];
        nodeType.prototype[hook] = function(...args2) {
          const result = previous?.apply(this, args2);
          ensureLoader(this);
          return result;
        };
      }
      const executed = nodeType.prototype.onExecuted;
      nodeType.prototype.onExecuted = function(...args2) {
        executed?.apply(this, args2);
        ensureLoader(this);
        const info = args2[0]?.video_info?.[0];
        if (info) loaders.get(this).info.value = infoText(info);
      };
      const removed = nodeType.prototype.onRemoved;
      nodeType.prototype.onRemoved = function(...args2) {
        const state = loaders.get(this);
        state?.request?.abort();
        state?.input.remove();
        loaders.delete(this);
        return removed?.apply(this, args2);
      };
    } else {
      const executed = nodeType.prototype.onExecuted;
      nodeType.prototype.onExecuted = function(...args2) {
        executed?.apply(this, args2);
        let root = players.get(this);
        if (!root) {
          root = document.createElement("div");
          root.style.cssText = "width:100%;height:100%;overflow:auto;display:flex;flex-direction:column;gap:6px";
          const widget = transient(this.addDOMWidget("video_preview", "alexz_video", root, { serialize: false }));
          widget.computeSize = (width) => [width, Math.max(260, (width || 400) * 0.75 + 80)];
          players.set(this, root);
        }
        root.replaceChildren();
        const message = args2[0];
        for (const item of message.alexz_videos ?? []) {
          const video = document.createElement("video");
          video.controls = true;
          video.preload = "metadata";
          video.style.cssText = "width:100%;max-height:calc(100% - 80px);object-fit:contain";
          video.src = `/view?${new URLSearchParams({ filename: item.filename, subfolder: item.subfolder, type: item.type })}`;
          root.append(video);
        }
        for (const item of message.alexz_files ?? []) {
          const link = document.createElement("a");
          link.href = `/view?${new URLSearchParams({ filename: item.filename, subfolder: item.subfolder, type: item.type })}`;
          link.download = item.filename;
          link.textContent = `\u0421\u043A\u0430\u0447\u0430\u0442\u044C ${item.filename}`;
          root.append(link);
        }
        const text = document.createElement("p");
        text.textContent = (message.text ?? []).join("\n");
        root.append(text);
        this.setSize([Math.max(this.size[0], 400), Math.max(this.computeSize()[1], 440)]);
      };
    }
  }
});
//# sourceMappingURL=video_io.js.map
