"""Keep source/camera checkpoints while releasing fast film intermediates after use."""

import functools
import logging
import sys

_LOGGER = logging.getLogger("SilentFilmMemory")
PERSISTENT_NODES = frozenset({"VideoLoadSource", "VideoSilentFilmCamera"})
TRANSIENT_NODES = frozenset({
    "VideoSilentFilmNegative", "VideoSilentFilmPrint", "VideoSilentFilmProjector",
    "VideoSilentFilmCaptureDirt", "VideoSilentFilmNegativeDevelopment", "VideoSilentFilmNegativeDamage",
    "VideoSilentFilmPrintingDirt", "VideoSilentFilmPositiveDevelopment", "VideoSilentFilmPositiveDamage",
    "VideoSilentFilmProjectorDirt", "VideoSilentFilmCadence", "VideoSilentFilmFinish", "VideoSaveFinal",
})


class FilmOutputCache:
    """Filter persistent storage only; let ExecutionList own live dependency references."""

    def __init__(self, inner):
        """Wrap one output cache without changing object caches or tensor storages."""
        self.inner = inner

    def __getattr__(self, name):
        """Preserve native eviction, providers, diagnostics and cache configuration."""
        return getattr(self.inner, name)

    def transient(self, node_id, visiting=None):
        """Propagate recomputation through consumers, stopping at explicit checkpoints."""
        graph = getattr(self.inner, "dynprompt", None)
        if graph is None or not graph.has_node(node_id):
            return False
        visiting = set() if visiting is None else visiting
        if node_id in visiting:
            return False
        visiting.add(node_id)
        node = graph.get_node(node_id)
        kind = node.get("class_type")
        if kind in PERSISTENT_NODES:
            return False
        if kind in TRANSIENT_NODES:
            return True
        # Иначе кэшированный PreviewImage/сборщик мог бы скрыть всю быструю цепочку.
        return any(self.transient(str(value[0]), visiting) for value in node.get("inputs", {}).values()
                   if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str) and isinstance(value[1], int))

    async def get(self, node_id):
        """Never reuse a fast stage result from a previous prompt."""
        if not self.transient(node_id):
            return await self.inner.get(node_id)
        return None

    def get_local(self, node_id):
        """Expose checkpoints to graph scheduling while live inputs stay in ExecutionList."""
        return None if self.transient(node_id) else self.inner.get_local(node_id)

    async def set(self, node_id, value):
        """Store only persistent outputs; transient values already have live consumers."""
        if not self.transient(node_id):
            await self.inner.set(node_id, value)

    def set_local(self, node_id, value):
        """Prevent ExecutionList.get_cache from writing transient inputs back on touch."""
        if not self.transient(node_id):
            self.inner.set_local(node_id, value)

    async def ensure_subcache_for(self, node_id, children_ids):
        """Apply the same policy to ephemeral subgraphs without touching core files."""
        child = await self.inner.ensure_subcache_for(node_id, children_ids)
        return self if child is self.inner else FilmOutputCache(child)


def enabled():
    """Report whether the executor constructor has the selective output adapter."""
    cache_set = getattr(sys.modules.get("execution"), "CacheSet", None)
    return bool(cache_set and getattr(cache_set.__init__, "_alexz_film_cache", False))


def install():
    """Install the narrow output-cache adapter against the inspected local executor API."""
    try:
        from execution import CacheSet, CacheType
        from comfy_execution.graph import ExecutionList
    except ImportError:
        return False
    original = CacheSet.__init__
    if getattr(original, "_alexz_film_cache", False):
        return True
    # Публичный CacheProvider добавляет внешнее хранилище, но не запрещает core-кэш.
    # Перехватываем только создание output cache; core-файлы и objects не изменяем.
    if not all(hasattr(ExecutionList, name) for name in ("cache_update", "get_cache", "complete_node_execution")):
        _LOGGER.warning("[Silent Film] Несовместимый исполнитель: выборочное кэширование не включено.")
        return False

    @functools.wraps(original)
    def initialize(self, *args, **kwargs):
        """Keep the selected core cache policy behind a film-specific output filter."""
        original(self, *args, **kwargs)
        cache_type = kwargs.get("cache_type", args[0] if args else None)
        if cache_type == CacheType.NONE:
            return
        if not hasattr(self, "outputs") or not hasattr(self, "all"):
            raise RuntimeError("Silent Film: интерфейс CacheSet изменился; требуется обновить адаптер выборочного кэша.")
        previous = self.outputs
        self.outputs = FilmOutputCache(previous)
        self.all = [self.outputs if cache is previous else cache for cache in self.all]

    initialize._alexz_film_cache = True
    CacheSet.__init__ = initialize
    _LOGGER.info("[Silent Film] Кэшируются Load Source Video и Camera; быстрые этапы освобождаются после потребителей.")
    return True
