import os

from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from neural_forge_ai import __version__, logger
from neural_forge_ai.app.api_routes import api_router
from neural_forge_ai.app.oauth import attach_oauth
from neural_forge_ai.app.ui_routes import ui_router


logger.info("Starting neural_forge_ai...")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = FastAPI()
if "SPACE_ID" in os.environ:
    attach_oauth(app)

app.include_router(ui_router, prefix="/ui", include_in_schema=False)
app.include_router(api_router, prefix="/api")
static_path = os.path.join(BASE_DIR, "static")
app.mount("/static", StaticFiles(directory=static_path), name="static")
logger.info(f"neural_forge_ai version: {__version__}")
logger.info("neural_forge_ai started successfully")


@app.get("/")
async def forward_to_ui(request: Request):
    """
    Forwards the incoming request to the UI endpoint.

    Args:
        request (Request): The incoming HTTP request.

    Returns:
        RedirectResponse: A response object that redirects to the UI endpoint,
                          including any query parameters from the original request.
    """
    query_params = request.query_params
    url = "/ui/"
    if query_params:
        url += f"?{query_params}"
    return RedirectResponse(url=url)
