"""Print the API's OpenAPI schema -- no server, no data needed.

    python -m recoup.api.openapi > web/openapi.json

``web/``'s ``npm run gen:api`` runs this and then openapi-typescript, so the
frontend's types always come from ``schemas.py``.
"""

import json
import sys
from pathlib import Path

from .app import create_app
from .service import Paths

if __name__ == "__main__":
    nowhere = Path("__no_data__")
    app = create_app(Paths(nowhere, nowhere, nowhere, nowhere, nowhere), static_dir=nowhere)
    json.dump(app.openapi(), sys.stdout, indent=1, ensure_ascii=False)
    sys.stdout.write("\n")
