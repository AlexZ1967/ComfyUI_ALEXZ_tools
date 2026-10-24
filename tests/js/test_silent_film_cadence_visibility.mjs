import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

const source = await readFile(new URL("../../web/widget_visibility_profiles.js", import.meta.url), "utf8");
const appImport = 'import { app } from "../../../scripts/app.js";';
assert.ok(source.includes(appImport));
let extension;
const app = { registerExtension(value) { extension = value; } };
// Загружаем настоящее расширение с подменой только объекта ComfyUI app.
new Function("app", source.replace(appImport, ""))(app);

class Node {
    constructor() {
        this.widgets = [
            { name: "source_fps", value: 24 },
            { name: "cadence_model", value: "legacy_random", callback() {} },
            { name: "crank_frames_per_turn", value: 12 },
            { name: "crank_irregularity", value: 0.15 },
            { name: "camera_shake_strength", value: 0.25 },
        ].map((widget) => ({ type: "number", computeSize: () => [100, 20], ...widget }));
        this.inputs = [];
        this.size = [300, 240];
        this.graph = { setDirtyCanvas() {} };
    }

    computeSize() { return [300, 240]; }
    setSize(size) { this.size = size; }
}

await extension.beforeRegisterNodeDef(Node, { name: "VideoSilentFilmCadence" });
const node = new Node();
node.onNodeCreated();
const widget = (name) => node.widgets.find((item) => item.name === name);
const names = node.widgets.map((item) => item.name);
const controller = widget("cadence_model");

for (const name of ["crank_frames_per_turn", "crank_irregularity", "camera_shake_strength"]) {
    assert.equal(widget(name).hidden, true);
}
controller.value = "hand_crank";
controller.callback();
assert.equal(widget("crank_frames_per_turn").hidden, true);
assert.notEqual(widget("crank_irregularity").hidden, true);
assert.notEqual(widget("camera_shake_strength").hidden, true);
assert.notEqual(widget("source_fps").hidden, true);
assert.equal(widget("crank_frames_per_turn").value, 12);
assert.deepEqual(node.widgets.map((item) => item.name), names);
assert.notEqual(widget("crank_frames_per_turn").serialize, false);

// Сохранение и повторная настройка не меняют порядок и значения виджетов.
node.onConfigure();
assert.equal(widget("crank_frames_per_turn").hidden, true);
assert.equal(widget("camera_shake_strength").value, 0.25);
controller.value = "legacy_random";
controller.callback();
assert.equal(widget("camera_shake_strength").hidden, true);
controller.value = "hand_crank";
controller.callback();
assert.notEqual(widget("camera_shake_strength").hidden, true);
assert.equal(widget("camera_shake_strength").value, 0.25);
console.log("Silent Film Cadence visibility: PASS");
