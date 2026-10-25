/** Browser-managed JSON saves without exposing arbitrary server filesystem paths. */

export interface Workflow {
    id?: string;
    nodes: unknown[];
    links: unknown[];
    extra?: Record<string, unknown>;
    [key: string]: unknown;
}
export interface WorkflowFileHandle {
    name: string;
    getFile(): Promise<{ lastModified: number; size: number }>;
    createWritable(): Promise<{
        write(data: string): Promise<void>;
        close(): Promise<void>;
        abort(): Promise<void>;
    }>;
}
export interface WorkflowPickerOptions {
    id: string;
    suggestedName: string;
    types: { description: string; accept: Record<string, string[]> }[];
    startIn?: WorkflowFileHandle;
}
export type WorkflowFilePicker = (options: WorkflowPickerOptions) => Promise<WorkflowFileHandle>;
interface Target { handle: WorkflowFileHandle; lastModified: number; size: number }
interface Options {
    graphId(): string;
    workflow(): Promise<Workflow>;
    suggestedName(): string;
    picker(): WorkflowFilePicker | undefined;
}

export class WorkflowFileSaver {
    private targets = new Map<string, Target>();
    private busy = new Set<string>();

    constructor(private options: Options) {}

    /** Save to an explicit native-picker destination or reuse this graph's chosen handle. */
    async save(saveAs = false): Promise<{ cancelled: boolean; filename?: string }> {
        const id = this.options.graphId();
        if (this.busy.has(id)) throw new Error("Сохранение этого workflow уже выполняется.");
        const previous = this.targets.get(id);
        const chooseFile = saveAs || !previous;
        const picker = this.options.picker();
        if (chooseFile && !picker) throw new Error("Браузер не поддерживает системный диалог сохранения. Используйте стандартный Export и включите в настройках браузера запрос папки для скачивания.");
        this.busy.add(id);
        try {
            let handle: WorkflowFileHandle;
            if (chooseFile) {
                try {
                    // Диалог вызывается до сериализации, пока действует пользовательское нажатие.
                    handle = await picker!({
                        id: "alexz-workflow-json",
                        suggestedName: previous?.handle.name ?? this.options.suggestedName(),
                        types: [{ description: "ComfyUI workflow (JSON)", accept: { "application/json": [".json"] } }],
                        ...(previous ? { startIn: previous.handle } : {}),
                    });
                } catch (error) {
                    if (typeof error === "object" && error !== null && "name" in error && error.name === "AbortError") return { cancelled: true };
                    throw error;
                }
            } else handle = previous!.handle;
            if (this.options.graphId() !== id) throw new Error("Активный workflow изменился. Повторите сохранение для нужного графа.");
            const workflow = await this.options.workflow();
            if (this.options.graphId() !== id) throw new Error("Активный workflow изменился. Повторите сохранение для нужного графа.");
            const text = JSON.stringify(workflow, null, 2) + "\n";
            if (!chooseFile && previous) {
                const current = await handle.getFile();
                if (current.lastModified !== previous.lastModified || current.size !== previous.size) throw new Error("Файл изменён другим приложением. Используйте «Сохранить как…» и выберите файл заново, если хотите его перезаписать.");
            }
            const writer = await handle.createWritable();
            try {
                await writer.write(text);
                await writer.close();
            } catch (error) {
                await writer.abort().catch(() => {});
                throw error;
            }
            const file = await handle.getFile();
            this.targets.set(id, { handle, lastModified: file.lastModified, size: file.size });
            return { cancelled: false, filename: handle.name };
        } finally { this.busy.delete(id); }
    }
}
