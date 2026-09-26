"""HTTP routes for the operator console. Thin by design: validate, call the
service, map its exceptions to status codes. The console itself (``web/``,
built to ``web/dist``) is served from ``/`` by the same process.

    recoup-ops serve                         # env paths, like the batch jobs
    recoup-ops serve --workdir var/demo      # the demo's layout
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import schemas as S
from .service import Conflict, NotReady, Paths, Workspace

ERRORS = {404: {"model": S.ApiError}, 409: {"model": S.ApiError}, 503: {"model": S.ApiError}}


def _default_static() -> Path:
    return Path(os.environ.get("RECOUP_WEB") or Path(__file__).resolve().parents[2] / "web" / "dist")


def create_app(paths: Paths | None = None, static_dir: Path | None = None) -> FastAPI:
    ws = Workspace(paths or Paths.from_env())
    app = FastAPI(title="recoup console API", version="0.3.0",
                  description="Read models over the recoup batch jobs' state. "
                              "Contract for web/ -- see recoup/api/schemas.py.")
    app.state.workspace = ws

    @app.exception_handler(NotReady)
    async def _not_ready(_: Request, e: NotReady):
        return JSONResponse(status_code=503, content={"detail": str(e)})

    @app.exception_handler(Conflict)
    async def _conflict(_: Request, e: Conflict):
        return JSONResponse(status_code=409, content={"detail": str(e)})

    # Plain ``def`` handlers: FastAPI runs them in a threadpool, which is what
    # the pandas-bound service wants.

    @app.get("/api/health", response_model=S.Health, tags=["meta"])
    def health():
        return ws.health()

    @app.get("/api/overview", response_model=S.Overview, responses=ERRORS, tags=["overview"])
    def overview():
        return ws.overview()

    @app.get("/api/gate", response_model=S.GateStatus, tags=["gate"])
    def gate():
        return ws.gate()

    @app.get("/api/gate/history", response_model=list[S.GateHistoryPoint], tags=["gate"])
    def gate_history():
        return ws.gate_history()

    @app.get("/api/model", response_model=Optional[S.ModelSummary], tags=["model"])
    def model():
        return ws.model()

    @app.get("/api/model/versions", response_model=list[S.ModelVersion], tags=["model"])
    def model_versions():
        return ws.model_versions()

    @app.get("/api/model/coefficients", response_model=list[S.Coefficient], responses=ERRORS,
             tags=["model"])
    def coefficients():
        return ws.coefficients()

    @app.get("/api/decisions", response_model=S.DecisionPage, responses=ERRORS,
             tags=["decisions"])
    def decisions(mode: Optional[str] = None, policy: Optional[str] = None,
                  action: Optional[str] = None, invoice_id: Optional[str] = None,
                  attempt_index: Optional[int] = None, offset: int = Query(0, ge=0),
                  limit: int = Query(50, ge=1, le=500)):
        return ws.decisions(mode=mode, policy=policy, action=action, invoice_id=invoice_id,
                            attempt_index=attempt_index, offset=offset, limit=limit)

    @app.get("/api/decisions/daily", response_model=list[S.DailyCount], responses=ERRORS,
             tags=["decisions"])
    def daily(days: int = Query(60, ge=1, le=730)):
        return ws.daily(days)

    @app.get("/api/decisions/coverage", response_model=S.Coverage, responses=ERRORS,
             tags=["decisions"])
    def coverage():
        return ws.coverage()

    @app.get("/api/invoices", response_model=S.InvoicePage, responses=ERRORS, tags=["invoices"])
    def invoices(status: Optional[str] = None, reason_class: Optional[str] = None,
                 rail: Optional[str] = None, q: Optional[str] = None,
                 offset: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=500)):
        return ws.invoices(status=status, reason_class=reason_class, rail=rail, q=q,
                           offset=offset, limit=limit)

    @app.get("/api/invoices/{invoice_id}", response_model=S.InvoiceDetail, responses=ERRORS,
             tags=["invoices"])
    def invoice(invoice_id: str, explain: bool = True):
        try:
            return ws.invoice(invoice_id, explain=explain)
        except KeyError:
            raise HTTPException(404, f"no invoice {invoice_id}")

    @app.get("/api/sim", response_model=S.SimStatus, tags=["simulation"])
    def sim_status():
        return ws.sim_status()

    @app.post("/api/sim/step", response_model=S.SimStatus, status_code=202, responses=ERRORS,
              tags=["simulation"])
    def sim_step(req: S.StepRequest):
        ws.sim_step(req)
        return ws.sim_status()

    @app.get("/api/sim/score", response_model=S.Score, responses=ERRORS, tags=["simulation"])
    def sim_score():
        return ws.score()

    # -- the console ------------------------------------------------------------
    static = static_dir or _default_static()
    if (static / "index.html").exists():
        if (static / "assets").exists():
            app.mount("/assets", StaticFiles(directory=static / "assets"), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str):
            if path.startswith("api/"):
                raise HTTPException(404)
            f = static / path
            if path and f.is_file() and static in f.resolve().parents:
                return FileResponse(f)
            return FileResponse(static / "index.html")

    return app


def app_from_env() -> FastAPI:
    """uvicorn factory: ``uvicorn --factory recoup.api.app:app_from_env``."""
    wd = os.environ.get("RECOUP_WORKDIR")
    return create_app(Paths.from_env(wd))
