"""OAuth support for AutoTrain.
Taken from: https://github.com/gradio-app/gradio/blob/main/gradio/oauth.py
"""

from __future__ import annotations

import hashlib
import os
import urllib.parse

import fastapi
from authlib.integrations.base_client.errors import MismatchingStateError
from authlib.integrations.starlette_client import OAuth
from fastapi.responses import RedirectResponse
from starlette.middleware.sessions import SessionMiddleware


OAUTH_CLIENT_ID = os.environ.get("OAUTH_CLIENT_ID")
OAUTH_CLIENT_SECRET = os.environ.get("OAUTH_CLIENT_SECRET")
OAUTH_SCOPES = os.environ.get("OAUTH_SCOPES")
OPENID_PROVIDER_URL = os.environ.get("OPENID_PROVIDER_URL")


def attach_oauth(app: fastapi.FastAPI):
    """
    Attaches OAuth functionality to a FastAPI application by adding OAuth routes and session middleware.

    Args:
        app (fastapi.FastAPI): The FastAPI application instance to which OAuth routes and middleware will be attached.

    Notes:
        - The session middleware requires a secret key to sign the cookies. A hash of the OAuth secret key is used to
          make it unique to the Space and to ensure it is updated if the OAuth configuration changes.
        - The session secret includes a version identifier ("-autotrain-v2") to allow for future changes in the session
          cookie format. If the format changes, the version can be bumped to invalidate old cookies and prevent HTTP 500 errors.
    """
    _add_oauth_routes(app)
    # Session Middleware requires a secret key to sign the cookies. Let's use a hash
    # of the OAuth secret key to make it unique to the Space + updated in case OAuth
    # config gets updated.
    session_secret = OAUTH_CLIENT_SECRET + "-autotrain-v2"
    # ^ if we change the session cookie format in the future, we can bump the version of the session secret to make
    #   sure cookies are invalidated. Otherwise some users with an old cookie format might get a HTTP 500 error.
    app.add_middleware(
        SessionMiddleware,
        secret_key=hashlib.sha256(session_secret.encode()).hexdigest(),
        https_only=True,
        same_site="none",
    )


def _add_oauth_routes(app: fastapi.FastAPI) -> None:
    """
    Add OAuth routes to the FastAPI app (login, callback handler, and logout).

    This function performs the following tasks:
    1. Checks for required environment variables and raises a ValueError if any are missing.
    2. Registers the OAuth server with the provided client ID, client secret, scopes, and OpenID provider URL.
    3. Defines the following OAuth routes:
        - `/login/huggingface`: Redirects to the Hugging Face OAuth page.
        - `/auth`: Handles the OAuth callback and manages the OAuth state.

    Args:
         app (fastapi.FastAPI): The FastAPI application instance to which the OAuth routes will be added.

    Raises:
         ValueError: If any of the required environment variables (OAUTH_CLIENT_ID, OAUTH_CLIENT_SECRET,
                         OAUTH_SCOPES, OPENID_PROVIDER_URL) are not set.
    """
    """Add OAuth routes to the FastAPI app (login, callback handler and logout)."""
    # Check environment variables
    msg = (
        "OAuth is required but {} environment variable is not set. Make sure you've enabled OAuth in your Space by"
        " setting `hf_oauth: true` in the Space metadata."
    )
    if OAUTH_CLIENT_ID is None:
        raise ValueError(msg.format("OAUTH_CLIENT_ID"))
    if OAUTH_CLIENT_SECRET is None:
        raise ValueError(msg.format("OAUTH_CLIENT_SECRET"))
    if OAUTH_SCOPES is None:
        raise ValueError(msg.format("OAUTH_SCOPES"))
    if OPENID_PROVIDER_URL is None:
        raise ValueError(msg.format("OPENID_PROVIDER_URL"))

    # Register OAuth server
    oauth = OAuth()
    oauth.register(
        name="huggingface",
        client_id=OAUTH_CLIENT_ID,
        client_secret=OAUTH_CLIENT_SECRET,
        client_kwargs={"scope": OAUTH_SCOPES},
        server_metadata_url=OPENID_PROVIDER_URL + "/.well-known/openid-configuration",
    )

    # Define OAuth routes
    @app.get("/login/huggingface")
    async def oauth_login(request: fastapi.Request):
        """
        Handles the OAuth login process by redirecting to the Hugging Face OAuth page.

        Args:
            request (fastapi.Request): The incoming HTTP request.

        Returns:
            Response: A redirection response to the Hugging Face OAuth authorization page.
        """
        """Endpoint that redirects to HF OAuth page."""
        redirect_uri = request.url_for("auth")
        redirect_uri_as_str = str(redirect_uri)
        if redirect_uri.netloc.endswith(".hf.space"):
            redirect_uri_as_str = redirect_uri_as_str.replace("http://", "https://")
        return await oauth.huggingface.authorize_redirect(request, redirect_uri_as_str)  # type: ignore

    @app.get("/auth")
    async def auth(request: fastapi.Request) -> RedirectResponse:
        """
        Handles the OAuth callback for Hugging Face authentication.

        Args:
            request (fastapi.Request): The incoming request object.

        Returns:
            RedirectResponse: A response object that redirects the user to the appropriate page.

        Raises:
            MismatchingStateError: If there is a state mismatch, likely due to a corrupted cookie.
                In this case, the user is redirected to the login page after clearing the relevant session keys.

        Notes:
            - If the state mismatch occurs, it is likely due to a bug in authlib that causes the token to grow indefinitely
              if the user tries to login repeatedly. Since cookies cannot exceed 4kb, the token will be truncated at some point,
              resulting in a lost state. The workaround is to delete the cookie and redirect the user to the login page again.
            - See https://github.com/lepture/authlib/issues/622 for more details.
        """
        """Endpoint that handles the OAuth callback."""
        try:
            oauth_info = await oauth.huggingface.authorize_access_token(request)  # type: ignore
        except MismatchingStateError:
            # If the state mismatch, it is very likely that the cookie is corrupted.
            # There is a bug reported in authlib that causes the token to grow indefinitely if the user tries to login
            # repeatedly. Since cookies cannot get bigger than 4kb, the token will be truncated at some point - hence
            # losing the state. A workaround is to delete the cookie and redirect the user to the login page again.
            # See https://github.com/lepture/authlib/issues/622 for more details.
            login_uri = "/login/huggingface"
            if "_target_url" in request.query_params:
                login_uri += "?" + urllib.parse.urlencode(  # Keep same _target_url as before
                    {"_target_url": request.query_params["_target_url"]}
                )
            for key in list(request.session.keys()):
                # Delete all keys that are related to the OAuth state
                if key.startswith("_state_huggingface"):
                    request.session.pop(key)
            return RedirectResponse(login_uri)

        request.session["oauth_info"] = oauth_info
        return _redirect_to_target(request)


def _redirect_to_target(request: fastapi.Request, default_target: str = "/") -> RedirectResponse:
    """
    Redirects the incoming request to a target URL specified in the query parameters.

    Args:
        request (fastapi.Request): The incoming HTTP request.
        default_target (str, optional): The default URL to redirect to if no target URL is specified in the query parameters. Defaults to "/".

    Returns:
        RedirectResponse: A response object that redirects the client to the target URL.
    """
    target = request.query_params.get("_target_url", default_target)
    # target = "https://huggingface.co/spaces/" + os.environ.get("SPACE_ID")
    return RedirectResponse(target)
