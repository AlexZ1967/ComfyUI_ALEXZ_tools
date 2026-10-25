import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { transform } from "esbuild";

const extensions = [], elements = [], requests = [];
const app = { registerExtension(extension) { extensions.push(extension); } };
const metadata = { total_frames: 6, fps: 24, width: 64, height: 48, codec: "prores", pixel_format: "yuv422p10le" };
const api = { async fetchApi(path, options) {
    requests.push({ path, options });
    return { ok: true, json: async () => path.includes("upload") ? { name: "Тест.mov", subfolder: "alexz_video" } : metadata };
} };
const document = {
    createElement(tag) {
        const element = { tag, style: {}, children: [], append(...items) { this.children.push(...items); }, replaceChildren() { this.children = []; }, remove() { this.removed = true; }, click() { this.clicked = true; } };
        elements.push(element); return element;
    },
    body: { append() {} },
};
const ComfyWidgets = { STRING(node, name) { const widget = { name, value: "", inputEl: {}, options: {} }; node.widgets.push(widget); return { widget }; } };
const source = await readFile(new URL("../../frontend/src/video_io.ts", import.meta.url), "utf8");
const compiled = await transform(source, { loader: "ts", target: "es2022", format: "esm" });
new Function("app", "api", "ComfyWidgets", "document", compiled.code.replace(/^import .*?;$/gm, ""))(app, api, ComfyWidgets, document);
const profiles = await readFile(new URL("../../web/widget_visibility_profiles.js", import.meta.url), "utf8");
new Function("app", profiles.replace(/^import .*?;$/gm, ""))(app);

class Node {
    constructor(values) {
        this.widgets = Object.entries(values).map(([name, value]) => ({ name, value, hidden: false, type: typeof value === "string" ? "combo" : "number", options: {}, computeSize: () => [430, 20] }));
        this.size = [430, 300]; this.inputs = []; this.graph = { links: {}, setDirtyCanvas() {} };
    }
    addWidget(type, name, value, callback) { const widget = { type, name, value, callback, options: {}, computeSize: () => [430, 20] }; this.widgets.push(widget); return widget; }
    addDOMWidget(name, type, element, options) { return this.addWidget(type, name, element, null, options); }
    setSize(size) { this.size = size; }
    computeSize() { return [430, 40 + this.widgets.reduce((sum, widget) => sum + (widget.hidden ? 0 : widget.computeSize?.(430)?.[1] ?? 60), 0)]; }
    removeInput(index) { this.inputs.splice(index, 1); }
}
class Loader extends Node {}
class Saver extends Node {}
for (const extension of extensions) {
    await extension.beforeRegisterNodeDef(Loader, { name: "VideoLoadSource" }, app);
    await extension.beforeRegisterNodeDef(Saver, { name: "VideoSaveFinal" }, app);
}
const load = new Loader({ video_path: "clip.mov", selection_mode: "full_video", frame_index: 2, start_frame: 1, end_frame: 4 });
load.onNodeCreated(); load.onConfigure();
const widget = name => load.widgets.find(widget => widget.name === name);
assert.deepEqual(load.widgets.filter(w => w.serialize !== false).map(w => w.name), ["video_path", "selection_mode", "frame_index", "start_frame", "end_frame"]);
assert.equal(load.widgets.length, 8);
for (const [mode, visible] of [["full_video", []], ["single_frame", ["frame_index"]], ["frame_range", ["start_frame", "end_frame"]]]) {
    widget("selection_mode").value = mode; widget("selection_mode").callback();
    assert.deepEqual(["frame_index", "start_frame", "end_frame"].filter(name => !widget(name).hidden), visible);
}
widget("Load and calculate frame").callback();
await new Promise(setImmediate);
assert.match(widget("video_info").value, /prores.*\/.*yuv422p10le/s);
assert.equal(requests[0].path, "/alexz/video/probe?path=clip.mov");
assert(!requests.some(request => request.path.includes("prompt")));
const input = elements.find(element => element.type === "file");
input.files = [new File(["data"], "Тест.mov")];
await input.onchange();
assert.equal(widget("video_path").value, "alexz_video/Тест.mov");
const upload = requests.find(request => request.options?.method === "POST");
assert.equal(upload.options.headers["X-Alexz-Video"], "1");
assert.equal(upload.options.body, input.files[0]);
load.onExecuted({ video_info: [{ ...metadata, start_frame: 1, end_frame: 3, selected_frames: 3 }] });
assert.match(widget("video_info").value, /Выбрано: 1–3/);
load.onRemoved(); assert(input.removed);

const save = new Saver({ frame_rate: 25, filename_prefix: "ALEXZ/film", format: "prores_422_hq", crf: 18, create_preview: true });
save.onNodeCreated();
const format = save.widgets.find(w => w.name === "format"), crf = save.widgets.find(w => w.name === "crf");
assert(crf.hidden);
format.value = "h264_mp4"; format.callback(); assert.equal(crf.hidden, false);
save.onExecuted({ alexz_videos: [{ filename: "preview.mp4", subfolder: "ALEXZ", type: "output" }], alexz_files: [{ filename: "master.mov", subfolder: "ALEXZ", type: "output" }], text: ["master.mov"] });
assert.equal(elements.find(element => element.tag === "video").src, "/view?filename=preview.mp4&subfolder=ALEXZ&type=output");
assert.equal(elements.find(element => element.tag === "a").download, "master.mov");
assert.equal(save.widgets.find(w => w.name === "video_preview").options.serialize, false);
format.value = "prores_4444"; format.callback();
assert(crf.hidden); assert(save.size[1] > 400);
console.log("Video IO upload, probe, selection, export controls and preview: PASS");
