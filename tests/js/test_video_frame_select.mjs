import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

const extensions = [];
const prompt = {
    output: {
        "1": { class_type: "VHS_LoadVideo", inputs: { video: "sample.mp4" } },
        "2": { class_type: "VideoFrameSelect", inputs: { image: ["1", 0], selection_mode: "frame_range", frame_index: ["99", 0], start_frame: 10, end_frame: 900 } },
        "3": { class_type: "VideoSilentFilmCamera", inputs: { image: ["2", 0] } },
        "4": { class_type: "PreviewImage", inputs: { images: ["3", 0] } },
        "99": { class_type: "UnrelatedHeavyNode", inputs: {} },
    },
    workflow: { nodes: [] },
};
const originalPrompt = JSON.stringify(prompt);
const app = { registerExtension(value) { extensions.push(value); }, async graphToPrompt() { return prompt; } };
const requests = [];
const api = { async queuePrompt(...args) { requests.push(args); return { prompt_id: "count" }; } };
const ComfyWidgets = {
    STRING(node, name) {
        const widget = { name, value: "", type: "text", inputEl: { readOnly: false }, computeSize: () => [300, 80] };
        node.widgets.push(widget);
        return { widget };
    },
};
for (const file of ["video_frame_select.js", "widget_visibility_profiles.js"]) {
    const source = await readFile(new URL(`../../web/${file}`, import.meta.url), "utf8");
    new Function("app", "api", "ComfyWidgets", source
        .replace('import { app } from "../../../scripts/app.js";', "")
        .replace('import { api } from "../../../scripts/api.js";', "")
        .replace('import { ComfyWidgets } from "../../../scripts/widgets.js";', ""))(app, api, ComfyWidgets);
}
let executed = 0;
class Node {
    constructor() {
        this.id = 2;
        this.size = [410, 330];
        this.widgets = [
            { name: "selection_mode", value: "full_video", type: "combo", callback() {} },
            ...["frame_index", "start_frame", "end_frame"].map(name => ({ name, value: 10, type: "number" })),
        ].map(w => ({ ...w, computeSize: () => [300, 20] }));
        this.inputs = [{ name: "image", link: 1 }, ...this.widgets.map(w => ({ name: w.name, widget: { name: w.name }, link: null }))];
        this.graph = { links: { 1: {} }, setDirtyCanvas() {} };
    }
    onExecuted() { executed += 1; }
    addWidget(type, name, value, callback, options = {}) {
        const w = { type, name, value, callback, options, computeSize: () => [300, 20] };
        this.widgets.push(w);
        return w;
    }
    removeInput(i) { this.inputs.splice(i, 1); }
    computeSize() { return [410, 330]; }
    setSize(value) { this.size = value; }
}
for (const extension of extensions) await extension.beforeRegisterNodeDef(Node, { name: "VideoFrameSelect" }, app);
const node = new Node();
node.onNodeCreated();
const widget = name => node.widgets.find(w => w.name === name);
assert.deepEqual(node.widgets.slice(0, 3).map(w => w.name), ["Load and calculate frame", "selection_info", "selection_mode"]);
const storedNames = node.widgets.filter(w => w.serialize !== false).map(w => w.name);
assert.deepEqual(storedNames, ["selection_mode", "frame_index", "start_frame", "end_frame"]);
function mode(value, visible) {
    widget("selection_mode").value = value;
    widget("selection_mode").callback();
    for (const name of ["frame_index", "start_frame", "end_frame"]) assert.equal(!widget(name).hidden, visible.includes(name), `${value}: ${name}`);
}
mode("full_video", []);
mode("single_frame", ["frame_index"]);
mode("frame_range", ["start_frame", "end_frame"]);
node.onConfigure();
assert.deepEqual(node.widgets.filter(w => w.serialize !== false).map(w => w.name), storedNames);
assert.equal(widget("frame_index").value, 10);
assert.equal(node._frameSelectionInfo.inputEl.readOnly, true);
for (const w of node.widgets.slice(0, 2)) {
    assert.equal(w.serialize, false);
    assert.equal(w.options.serialize, false);
}
await node._frameCountButton.callback();
assert.equal(requests.length, 1);
const [number, request, options] = requests[0];
assert.equal(number, 0);
assert.deepEqual(Object.keys(request.output).sort(), ["1", "2"]);
assert.deepEqual(request.output["2"].inputs, { image: ["1", 0], selection_mode: "full_video", frame_index: 0, start_frame: 0, end_frame: -1 });
assert.deepEqual(options, { partialExecutionTargets: ["2"] });
assert.equal(JSON.stringify(prompt), originalPrompt);
assert.equal(widget("selection_mode").value, "frame_range");
node.onExecuted({ video_info: [{ total_frames: 241, width: 1664, height: 1248 }] });
assert.equal(executed, 1);
assert.match(node._frameSelectionInfo.value, /241/);
assert.match(node._frameSelectionInfo.value, /1664 × 1248/);
assert.match(node._frameSelectionInfo.value, /0–240.*120/);
node.onExecuted({ text: ["В ролике: 99 кадров."] });
assert.equal(node._frameSelectionInfo.value, "В ролике: 99 кадров.");
assert.equal(node.widgets.length, 6);
const warn = console.warn;
try {
    console.warn = () => {};
    api.queuePrompt = async () => { throw new Error("Test failure"); };
    await node._frameCountButton.callback();
    assert.match(node._frameSelectionInfo.value, /Ошибка: Test failure/);
    assert.equal(node._frameCountLoading, false);
} finally { console.warn = warn; }
class Other {}
for (const extension of extensions) await extension.beforeRegisterNodeDef(Other, { name: "OtherNode" }, app);
assert.equal(Other.prototype.onExecuted, undefined);
console.log("Video Frame Select controls and count-only request: PASS");
