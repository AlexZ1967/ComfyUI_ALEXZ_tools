declare module "*scripts/app.js" {
  export const app: {
    registerExtension(extension: {
      name: string;
      init?: (...args: unknown[]) => unknown | Promise<unknown>;
      setup?: (...args: unknown[]) => unknown | Promise<unknown>;
      beforeRegisterNodeDef?: (...args: unknown[]) => unknown | Promise<unknown>;
      nodeCreated?: (...args: unknown[]) => unknown | Promise<unknown>;
      loadedGraphNode?: (...args: unknown[]) => unknown | Promise<unknown>;
      beforeConfigureGraph?: (...args: unknown[]) => unknown | Promise<unknown>;
      afterConfigureGraph?: (...args: unknown[]) => unknown | Promise<unknown>;
      [key: string]: unknown;
    }): void;
  };
}

declare module "*scripts/api.js" {
  export const api: {
    fetchApi(
      route: string,
      options?: RequestInit
    ): Promise<Response>;

    [key: string]: unknown;
  };
}
