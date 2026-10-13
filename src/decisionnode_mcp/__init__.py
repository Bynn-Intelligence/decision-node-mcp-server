"""The DecisionNode MCP server: DecisionNode's judgments as tools for AI agents."""

import warnings

__version__ = "0.1.1"

# A dependency of FastMCP warns about its own deprecated modules on import, and turns its
# warnings on when its warning module loads. The server does not use those modules, so
# the warning is switched off after that module has loaded, to keep the startup log clean.
try:
    from authlib.deprecate import AuthlibDeprecationWarning
except ImportError:  # pragma: no cover - not every FastMCP 2.x release pulls it in
    pass
else:
    warnings.filterwarnings("ignore", category=AuthlibDeprecationWarning)
