import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";
import { ComfyWidgets } from "../../../scripts/widgets.js";

interface Widget {
    name: string; value: unknown; serialize?: boolean; options?: Record<string, unknown>;
    inputEl?: HTMLTextAreaElement; callback?: (...args: unknown[]) => unknown;
    computeSize?: (width: number) => [number, number];
}
interface VideoNode {
    widgets: Widget[]; size: [number, number];
    graph?: { setDirtyCanvas(foreground: boolean, background: boolean): void };
    addWidget(type: string, name: string, value: unknown, callback: () => void): Widget;
    addDOMWidget(name: string, type: string, element: HTMLElement, options: Record<string, unknown>): Widget;
    setSize(size: [number, number]): void; computeSize(): [number, number];
}
type Hook = (this: VideoNode, ...args: unknown[]) => unknown;
interface NodeType { prototype: Record<string, Hook | undefined> }
interface VideoInfo { total_frames: number; fps: number; width: number; height: number; codec: string; pixel_format: string; selected_frames?: number; start_frame?: number; end_frame?: number }
interface VideoFile { filename: string; subfolder: string; type: string }

/** Keep helper controls out of serialized backend inputs and widget values. */
function transient(widget: Widget): Widget {
    widget.serialize = false;
    widget.options ??= {};
    widget.options.serialize = false;
    return widget;
}

/** Render source metadata without queueing the workflow or invoking downstream nodes. */
function infoText(info: VideoInfo): string {
    return `${info.codec} / ${info.pixel_format}\n${info.width} × ${info.height}; ${info.fps.toFixed(6)} fps\nКадров: ${info.total_frames}; номера 0–${info.total_frames - 1}`
        + (info.selected_frames ? `\nВыбрано: ${info.start_frame}–${info.end_frame}, кадров: ${info.selected_frames}` : "");
}

const loaders = new WeakMap<VideoNode, { input: HTMLInputElement; info: Widget; button: Widget; choose: Widget; request?: AbortController; busy: boolean }>();
const players = new WeakMap<VideoNode, HTMLElement>();

/** Add upload, metadata probe, and readonly information above selection controls. */
function ensureLoader(node: VideoNode): void {
    let state = loaders.get(node);
    if (!state) {
        const info = transient((ComfyWidgets as { STRING(node: VideoNode, name: string, input: unknown, app: unknown): { widget: Widget } })
            .STRING(node, "video_info", ["STRING", { multiline: true }], app).widget);
        if (info.inputEl) info.inputEl.readOnly = true;
        info.value = "Укажите путь к видео или выберите файл. Затем Load and calculate frame.";
        const input = document.createElement("input");
        input.type = "file";
        input.accept = "video/*,.mov,.mxf,.mkv,.avi,.mp4,.m4v,.webm,.mpg,.mpeg,.ts";
        input.style.display = "none";
        document.body.append(input);
        const choose = transient(node.addWidget("button", "Выбрать видео…", null, () => input.click()));
        const button = transient(node.addWidget("button", "Load and calculate frame", null, () => { void probe(); }));
        state = { input, info, choose, button, busy: false };
        loaders.set(node, state);
        const current = state;
        const pathWidget = () => node.widgets.find(widget => widget.name === "video_path");
        const changed = pathWidget()?.callback;
        const path = pathWidget();
        if (path) path.callback = function (...args) {
            current.request?.abort();
            current.info.value = "Файл изменён. Нажмите Load and calculate frame.";
            return changed?.apply(this, args);
        };
        async function probe(): Promise<void> {
            current.request?.abort();
            const controller = new AbortController();
            current.request = controller;
            const path = String(pathWidget()?.value ?? "");
            current.info.value = "Чтение информации… Если в контейнере нет числа кадров, выполняется подсчёт без сохранения изображений.";
            try {
                const response = await api.fetchApi(`/alexz/video/probe?path=${encodeURIComponent(path)}`, { signal: controller.signal });
                const data = await response.json();
                if (!response.ok) throw Error(response.status === 404 ? "Перезапустите ComfyUI для загрузки маршрута информации о видео." : data.error ?? `HTTP ${response.status}`);
                if (path === String(pathWidget()?.value ?? "")) current.info.value = infoText(data as VideoInfo);
            } catch (error) {
                if (!controller.signal.aborted) current.info.value = String(error);
            }
            node.graph?.setDirtyCanvas(true, true);
        }
        input.onchange = async () => {
            if (!input.files?.[0] || current.busy) return;
            current.busy = true;
            current.info.value = "Загрузка файла в ComfyUI/input…";
            try {
                const file = input.files[0];
                const response = await api.fetchApi(`/alexz/video/upload?filename=${encodeURIComponent(file.name)}`, {
                    method: "POST", headers: { "Content-Type": "application/octet-stream", "X-Alexz-Video": "1" }, body: file,
                });
                const data = await response.json();
                if (!response.ok || !data.name) throw Error(data.error ?? `Загрузка: HTTP ${response.status}`);
                const widget = pathWidget();
                if (widget) {
                    widget.value = [data.subfolder, data.name].filter(Boolean).join("/");
                    widget.callback?.(widget.value);
                }
                await probe();
            } catch (error) { current.info.value = String(error); }
            finally { current.busy = false; input.value = ""; }
        };
    }
    node.widgets = [state.choose, state.button, state.info, ...node.widgets.filter(widget => ![state.choose, state.button, state.info].includes(widget))];
}

app.registerExtension({
    name: "ALEXZ.Tools.VideoIO",
    beforeRegisterNodeDef(...args: unknown[]) {
        const nodeType = args[0] as NodeType;
        const name = (args[1] as { name: string }).name;
        if (!["VideoLoadSource", "VideoSaveFinal"].includes(name)) return;
        if (name === "VideoLoadSource") {
            for (const hook of ["onNodeCreated", "onConfigure"]) {
                const previous = nodeType.prototype[hook];
                nodeType.prototype[hook] = function (...args) { const result = previous?.apply(this, args); ensureLoader(this); return result; };
            }
            const executed = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (...args) {
                executed?.apply(this, args); ensureLoader(this);
                const info = (args[0] as { video_info?: VideoInfo[] })?.video_info?.[0];
                if (info) loaders.get(this)!.info.value = infoText(info);
            };
            const removed = nodeType.prototype.onRemoved;
            nodeType.prototype.onRemoved = function (...args) {
                const state = loaders.get(this); state?.request?.abort(); state?.input.remove(); loaders.delete(this);
                return removed?.apply(this, args);
            };
        } else {
            const executed = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (...args) {
                executed?.apply(this, args);
                let root = players.get(this);
                if (!root) {
                    root = document.createElement("div");
                    root.style.cssText = "width:100%;height:100%;overflow:auto;display:flex;flex-direction:column;gap:6px";
                    const widget = transient(this.addDOMWidget("video_preview", "alexz_video", root, { serialize: false }));
                    widget.computeSize = width => [width, Math.max(260, (width || 400) * 0.75 + 80)];
                    players.set(this, root);
                }
                root.replaceChildren();
                const message = args[0] as { alexz_videos?: VideoFile[]; alexz_files?: VideoFile[]; text?: string[] };
                for (const item of message.alexz_videos ?? []) {
                    const video = document.createElement("video");
                    video.controls = true; video.preload = "metadata"; video.style.cssText = "width:100%;max-height:calc(100% - 80px);object-fit:contain";
                    video.src = `/view?${new URLSearchParams({ filename: item.filename, subfolder: item.subfolder, type: item.type })}`;
                    root.append(video);
                }
                for (const item of message.alexz_files ?? []) {
                    const link = document.createElement("a");
                    link.href = `/view?${new URLSearchParams({ filename: item.filename, subfolder: item.subfolder, type: item.type })}`;
                    link.download = item.filename; link.textContent = `Скачать ${item.filename}`; root.append(link);
                }
                const text = document.createElement("p"); text.textContent = (message.text ?? []).join("\n"); root.append(text);
                this.setSize([Math.max(this.size[0], 400), Math.max(this.computeSize()[1], 440)]);
            };
        }
    },
});
