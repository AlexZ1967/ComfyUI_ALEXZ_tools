import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";
import { ComfyWidgets } from "../../../scripts/widgets.js";

// В запрос подсчёта включаются только селектор и его входные зависимости.
function countPrompt(output, nodeId) {
    const root = output[nodeId];
    if (!root || root.class_type !== "VideoFrameSelect") throw new Error("Селектор отсутствует в графе расчёта.");
    const source = {
        ...output,
        [nodeId]: { ...root, inputs: { image: root.inputs.image, selection_mode: "full_video", frame_index: 0, start_frame: 0, end_frame: -1 } },
    };
    const selected = {};
    function visit(id) {
        if (selected[id]) return;
        const node = source[id];
        if (!node) throw new Error(`Не найдена входная нода ${id}.`);
        selected[id] = node;
        for (const value of Object.values(node.inputs ?? {})) {
            if (Array.isArray(value) && value.length === 2 && typeof value[0] === "string" && Number.isInteger(value[1])) visit(value[0]);
        }
    }
    if (!Array.isArray(root.inputs.image)) throw new Error("Подключите image от загрузчика видео.");
    visit(nodeId);
    return selected;
}

app.registerExtension({
    name: "ALEXZ.Tools.VideoFrameSelect",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "VideoFrameSelect") return;

        function ensureControls(node) {
            if (!node._frameSelectionInfo) {
                const info = ComfyWidgets.STRING(node, "selection_info", ["STRING", { multiline: true }], app).widget;
                info.inputEl.readOnly = true;
                info.value = "Число кадров появится после Load and calculate frame.";
                const button = node.addWidget("button", "Load and calculate frame", null, async () => {
                    if (node._frameCountLoading) return;
                    node._frameCountLoading = true;
                    const previousInfo = info.value;
                    info.value = "Загрузка и подсчёт кадров… Остальные этапы не запускаются.";
                    try {
                        const prompt = await app.graphToPrompt();
                        const id = String(node.id);
                        const output = countPrompt(prompt.output, id);
                        await api.queuePrompt(0, { output, workflow: prompt.workflow }, { partialExecutionTargets: [id] });
                    } catch (error) {
                        info.value = `Ошибка: ${error?.message ?? error}\n${previousInfo}`;
                        console.warn("[ALEXZ_tools] Подсчёт кадров не запущен:", error);
                    } finally {
                        node._frameCountLoading = false;
                        node.graph?.setDirtyCanvas(true, true);
                    }
                });
                for (const widget of [button, info]) {
                    widget.serialize = false;
                    widget.options ??= {};
                    widget.options.serialize = false;
                }
                node._frameSelectionInfo = info;
                node._frameCountButton = button;
            }
            const { _frameCountButton: button, _frameSelectionInfo: info } = node;
            node.widgets = [button, info, ...node.widgets.filter(w => w !== button && w !== info)];
        }

        for (const hook of ["onNodeCreated", "onConfigure"]) {
            const original = nodeType.prototype[hook];
            nodeType.prototype[hook] = function () {
                const result = original?.apply(this, arguments);
                ensureControls(this);
                return result;
            };
        }
        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, arguments);
            ensureControls(this);
            const data = message?.video_info?.[0];
            this._frameSelectionInfo.value = data
                ? `Кадров в ролике: ${data.total_frames}.\nРазмер: ${data.width} × ${data.height}.\nНомера: 0–${data.total_frames - 1}; средний кадр: ${Math.floor(data.total_frames / 2)}.`
                : Array.isArray(message?.text) ? message.text.join("\n") : String(message?.text ?? "");
            this.setSize?.(this.computeSize());
            this.graph?.setDirtyCanvas(true, true);
        };
    },
});
