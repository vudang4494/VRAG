"""Plugin registry — discovers and loads all source plugins."""
import importlib
import pkgutil
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from plugins.base import BaseSourcePlugin, PluginCapability

if TYPE_CHECKING:
    from plugins.base import PluginConfig, SourceCredentials


class PluginRegistry:
    """
    Discovers and manages all source plugins.
    Plugins are found in plugins/sources/ and plugins/rerankers/.
    """

    _instance: "PluginRegistry | None" = None
    _sources: dict[str, type[BaseSourcePlugin]] = {}
    _initialized: bool = False

    def __new__(cls) -> "PluginRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def discover(self) -> None:
        """Scan plugin directories and register all plugins."""
        if self._initialized:
            return

        plugins_root = Path(__file__).parent
        sources_dir = plugins_root / "sources"

        # Discover source plugins
        for importer, modname, ispkg in pkgutil.iter_modules([str(sources_dir)]):
            if ispkg:
                try:
                    mod = importlib.import_module(f"plugins.sources.{modname}.plugin")
                    for attr_name in dir(mod):
                        cls = getattr(mod, attr_name)
                        if (
                            isinstance(cls, type)
                            and issubclass(cls, BaseSourcePlugin)
                            and cls is not BaseSourcePlugin
                        ):
                            self._sources[cls.name] = cls
                            logger.info(f"Registered plugin: {cls.name} v{cls.version}")
                except ImportError as e:
                    logger.warning(f"Could not load plugin {modname}: {e}")

        self._initialized = True
        logger.info(f"Plugin registry: {len(self._sources)} source plugins loaded")

    def get_source_plugin(self, name: str) -> type[BaseSourcePlugin] | None:
        """Get a source plugin class by name."""
        self.discover()
        return self._sources.get(name)

    def list_source_plugins(self) -> list[dict]:
        """List all registered source plugins with metadata."""
        self.discover()
        return [
            {
                "name": cls.name,
                "version": cls.version,
                "capabilities": [c.value for c in cls.capabilities],
                "supported_types": cls.supported_types,
            }
            for cls in self._sources.values()
        ]

    def create_source_plugin(
        self,
        name: str,
        config: "PluginConfig",
        credentials: "SourceCredentials | None" = None,
    ) -> BaseSourcePlugin:
        """Instantiate a source plugin with configuration."""
        cls = self.get_source_plugin(name)
        if cls is None:
            raise ValueError(f"Unknown plugin: {name}. Available: {list(self._sources.keys())}")
        return cls(config=config, credentials=credentials)

    @property
    def available_sources(self) -> list[str]:
        """Names of all available source plugins."""
        self.discover()
        return list(self._sources.keys())


# Global singleton
registry = PluginRegistry()
