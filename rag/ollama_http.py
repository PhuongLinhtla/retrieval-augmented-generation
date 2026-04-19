from __future__ import annotations

import base64
import json
from typing import Any
from urllib import error, request


def _join_url(base_url: str, path: str) -> str:
    root = (base_url or "http://127.0.0.1:11434").strip().rstrip("/")
    endpoint = path.strip()
    if not endpoint.startswith("/"):
        endpoint = f"/{endpoint}"
    return f"{root}{endpoint}"


def _post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url=url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama HTTP error {exc.code}: {detail}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Cannot connect to Ollama at {url}: {exc.reason}") from exc

    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON from Ollama endpoint {url}") from exc


def generate_text(
    host: str,
    model: str,
    prompt: str,
    *,
    num_ctx: int,
    temperature: float,
    timeout: float,
) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_ctx": max(512, int(num_ctx)),
            "temperature": max(0.0, float(temperature)),
        },
    }

    data = _post_json(_join_url(host, "/api/generate"), payload, timeout)
    text = str(data.get("response", "") or "").strip()
    if not text:
        raise RuntimeError("Ollama returned an empty response")
    return text


def embed_text(
    host: str,
    model: str,
    text: str,
    *,
    timeout: float,
) -> list[float]:
    payload = {
        "model": model,
        "prompt": text,
    }
    data = _post_json(_join_url(host, "/api/embeddings"), payload, timeout)

    vector = data.get("embedding")
    if not isinstance(vector, list) or not vector:
        raise RuntimeError("Ollama embedding endpoint returned invalid vector")

    out: list[float] = []
    for item in vector:
        try:
            out.append(float(item))
        except (TypeError, ValueError) as exc:
            raise RuntimeError("Ollama embedding vector contains non-numeric values") from exc
    return out


def vision_to_text(
    host: str,
    model: str,
    image_bytes: bytes,
    *,
    timeout: float,
    prompt: str | None = None,
) -> str:
    encoded = base64.b64encode(image_bytes).decode("ascii")
    instruction = prompt or (
        "Hay doc noi dung trong anh bang tieng Viet. "
        "Neu la bieu do/so do, hay tom tat cac diem chinh, nhan de, truc, don vi va xu huong."
    )

    payload = {
        "model": model,
        "prompt": instruction,
        "images": [encoded],
        "stream": False,
    }

    data = _post_json(_join_url(host, "/api/generate"), payload, timeout)
    text = str(data.get("response", "") or "").strip()
    if not text:
        raise RuntimeError("Ollama vision endpoint returned an empty response")
    return text
