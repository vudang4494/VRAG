"""V3 API router — backward-compat shim.

All endpoint handlers have been moved to api/routes/*.py.
This file re-exports the combined router so existing imports
like `from api.routes_v3 import router` continue to work.

New-style import (preferred):
    from api.routes import router

Legacy import (still works):
    from api.routes_v3 import router
"""

from api.routes import router

__all__ = ["router"]
