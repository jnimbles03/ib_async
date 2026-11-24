"""Spec for the Flex Web Service credential embedded in the statement module.

These tests fail while that credential is still in the package. They pass once
the program entry no longer transmits a digit-only token or query id, and the
package source no longer contains that credential.

The program entry is executed with a stand-in socket so the real download path
builds the HTTP request without contacting Interactive Brokers. Request
parameters are never printed.
"""

from __future__ import annotations

import ast
import http.client
import io
import os
import re
import runpy
import sys
from collections.abc import Iterable
from pathlib import Path
from urllib.parse import parse_qs, urlsplit
from xml.etree import ElementTree as et

import ib_async.flexreport as flexreport

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "ib_async"
MODULE_PATH = PACKAGE_ROOT / "flexreport.py"
SOURCE_ROOTS = (
    PACKAGE_ROOT,
    REPO_ROOT / "examples",
    REPO_ROOT / "docs",
    REPO_ROOT / "tests",
)
TEXT_SUFFIXES = {".py", ".md", ".rst", ".txt"}
# Digit-only values at least this long are treated as embedded Flex credentials
# when they are passed into the statement download. Obvious placeholders are
# words such as PLACEHOLDER_TOKEN and PLACEHOLDER_QUERY_ID.
MIN_EMBEDDED_DIGIT_LENGTH = 5
# A contiguous run of this many digits anywhere in shipped text is a Flex token.
MIN_TOKEN_RUN_LENGTH = 16
TOKEN_RUN = re.compile(rf"\d{{{MIN_TOKEN_RUN_LENGTH},}}")
DIGIT_SEQUENCE = re.compile(r"\d{5,}")


def _sanitize(text: str) -> str:
    return DIGIT_SEQUENCE.sub("[digit-sequence]", text)


def _iter_source_files(roots: Iterable[Path]) -> Iterable[Path]:
    for root in roots:
        if not root.exists():
            continue
        if root.is_file():
            yield root
            continue
        for path in root.rglob("*"):
            if (
                path.is_file()
                and path.suffix in TEXT_SUFFIXES
                and "notebooks" not in path.parts
            ):
                yield path


def _const_text(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str | int):
        return str(node.value)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _const_text(node.left)
        right = _const_text(node.right)
        if left is not None and right is not None:
            return left + right
    return None


def _is_digit_credential(value: str) -> bool:
    return len(value) >= MIN_EMBEDDED_DIGIT_LENGTH and value.isdigit()


def _assigned_constants(tree: ast.AST) -> dict[str, str]:
    assigned: dict[str, str] = {}
    for node in ast.walk(tree):
        target = None
        value = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            value = node.value
        elif isinstance(node, ast.AnnAssign):
            target = node.target
            value = node.value
        if isinstance(target, ast.Name) and value is not None:
            text = _const_text(value)
            if text is not None:
                assigned[target.id] = text
    return assigned


def _call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def hardcoded_flex_call_sites(path: Path) -> list[tuple[int, str]]:
    """Return file locations where a Flex download is given a digit-only credential."""
    if path.suffix != ".py":
        return []
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:
        return []
    assigned = _assigned_constants(tree)
    findings: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node.func)
        if name not in {"FlexReport", "download"}:
            continue
        expressions = list(node.args)
        expressions.extend(keyword.value for keyword in node.keywords)
        for expression in expressions:
            if isinstance(expression, ast.Name):
                text = assigned.get(expression.id)
            else:
                text = _const_text(expression)
            if text is not None and _is_digit_credential(text):
                findings.append((node.lineno, path.name))
                break
    return findings


def long_digit_runs(path: Path) -> list[int]:
    """Line numbers that contain a Flex-token-length digit run. Values are not returned."""
    lines: list[int] = []
    text = path.read_text(encoding="utf-8", errors="replace")
    for lineno, line in enumerate(text.splitlines(), 1):
        if TOKEN_RUN.search(line):
            lines.append(lineno)
    return lines


def _http_response(body: bytes) -> bytes:
    return (
        b"HTTP/1.1 200 OK\r\n"
        b"Content-Type: application/xml\r\n"
        b"Content-Length: " + str(len(body)).encode("ascii") + b"\r\n"
        b"Connection: close\r\n"
        b"\r\n" + body
    )


def _statement_ready_xml(statement_url: str) -> bytes:
    root = et.Element("FlexStatementResponse")
    et.SubElement(root, "Status").text = "Success"
    et.SubElement(root, "ReferenceCode").text = "PLACEHOLDER_CODE"
    et.SubElement(root, "Url").text = statement_url
    return et.tostring(root)


def _finished_statement_xml() -> bytes:
    root = et.Element("FlexQueryResponse")
    et.SubElement(root, "FlexStatements")
    return et.tostring(root)


class _RecordingSocket:
    """Stand-in for the TCP socket only. urllib and http.client still build the request."""

    def __init__(self, response_body: bytes, sink: list[bytes]):
        self._reader = io.BytesIO(_http_response(response_body))
        self._sink = sink

    def sendall(self, data):
        if isinstance(data, str):
            payload = data.encode("latin-1")
        else:
            payload = bytes(data)
        self._sink.append(payload)

    def makefile(self, mode="rb", *args, **kwargs):
        return self._reader

    def close(self):
        return None

    def setsockopt(self, *args, **kwargs):
        return None


def _parse_http_request(raw: bytes) -> tuple[str, str, str, dict[str, str]]:
    head = raw.split(b"\r\n\r\n", 1)[0]
    lines = head.split(b"\r\n")
    if not lines or b" " not in lines[0]:
        raise AssertionError("flex download did not build an HTTP request line")
    request_line = lines[0].decode("ascii", errors="replace")
    method, target, *_rest = request_line.split(" ")
    headers: dict[str, str] = {}
    for line in lines[1:]:
        decoded = line.decode("ascii", errors="replace")
        if ":" not in decoded:
            continue
        name, value = decoded.split(":", 1)
        headers[name.strip().lower()] = value.strip()
    if "://" in target:
        parts = urlsplit(target)
        host = parts.netloc
        path = parts.path
        query = parts.query
    else:
        parts = urlsplit(
            "https://flex.invalid"
            + (target if target.startswith("/") else "/" + target)
        )
        host = headers.get("host", "")
        path = parts.path
        query = parts.query
    parsed = {
        key: values[0]
        for key, values in parse_qs(query, keep_blank_values=True).items()
        if values
    }
    return method, host, path, parsed


def _run_program_entry(module_path: Path) -> list[dict[str, object]]:
    """Run the statement module as a program and return request metadata, never parameter values."""
    service = urlsplit(flexreport.FLEXREPORT_URL)
    statement_url = f"{service.scheme}://{service.netloc}/AccountManagement/FlexWebService/GetStatement"
    bodies = [
        _statement_ready_xml(statement_url),
        _finished_statement_xml(),
    ]
    chunks: list[bytes] = []
    state = {"index": 0}

    def fake_connect(self):
        index = state["index"]
        state["index"] = index + 1
        body = bodies[index] if index < len(bodies) else bodies[-1]
        self.sock = _RecordingSocket(body, chunks)

    import time

    original_sleep = time.sleep
    time.sleep = lambda _seconds: None
    previous_connect = http.client.HTTPSConnection.connect
    previous_http_connect = http.client.HTTPConnection.connect
    http.client.HTTPSConnection.connect = fake_connect
    http.client.HTTPConnection.connect = fake_connect
    removed_env = {}
    for name in list(os.environ):
        if (
            "FLEX" in name.upper()
            or name.lower().endswith("_proxy")
            or name.lower() == "all_proxy"
        ):
            removed_env[name] = os.environ.pop(name)
    try:
        try:
            runpy.run_path(str(module_path), run_name="__main__")
        except flexreport.FlexError:
            pass
        except Exception as exc:
            raise AssertionError(
                "flex program entry failed before the credential check: "
                + _sanitize(str(exc))
            ) from None
    finally:
        time.sleep = original_sleep
        http.client.HTTPSConnection.connect = previous_connect
        http.client.HTTPConnection.connect = previous_http_connect
        os.environ.update(removed_env)

    requests = []
    for raw in chunks:
        method, host, path, query = _parse_http_request(raw)
        requests.append(
            {
                "method": method,
                "host": host.split(":")[0],
                "path": path,
                "digit_credentials": sorted(
                    key
                    for key, value in query.items()
                    if key in {"t", "q"} and _is_digit_credential(value)
                ),
            }
        )
    return requests


def test_program_entry_transmits_no_hardcoded_flex_credential():
    """Running the statement module must not send a digit-only token or query id.

    Interactive Brokers' Flex download builds ``t=<token>&q=<query id>`` and
    hands that URL to ``urlopen``. Both the initial request and the follow-up
    statement request are covered. A fix replaces those literals with
    non-numeric placeholders or reads them from the environment, so this entry
    point no longer transmits a hardcoded credential.
    """
    requests = _run_program_entry(MODULE_PATH)
    leaked = [request for request in requests if request["digit_credentials"]]
    if leaked:
        summary = ", ".join(
            f"{request['method']} {request['path']} parameters {request['digit_credentials']}"
            for request in leaked
        )
        raise AssertionError(
            "Executing ib_async/flexreport.py as a program transmitted a hardcoded "
            "Flex Web Service credential. Replace the token and query id in the "
            "program entry with non-numeric placeholders, or load them from the "
            f"environment. Offending request metadata: {summary}"
        )


def test_package_source_has_no_hardcoded_flex_credential():
    """The shipped package must not still contain the Flex token or query id."""
    runs: list[str] = []
    calls: list[str] = []
    for path in _iter_source_files(SOURCE_ROOTS):
        for lineno in long_digit_runs(path):
            runs.append(f"{path.relative_to(REPO_ROOT)}:{lineno}")
        for lineno, _name in hardcoded_flex_call_sites(path):
            calls.append(f"{path.relative_to(REPO_ROOT)}:{lineno}")
    if runs or calls:
        raise AssertionError(
            "Remove the Flex Web Service token and query id from the package source "
            "and replace them with non-numeric placeholders such as PLACEHOLDER_TOKEN "
            "and PLACEHOLDER_QUERY_ID. "
            f"Long digit runs: {runs or 'none'}. "
            f"Flex calls given a digit-only credential: {calls or 'none'}."
        )


def test_flex_module_under_test_is_the_shipped_module():
    """The file executed above is the module the package imports."""
    shipped = Path(flexreport.__file__).resolve()
    assert shipped == MODULE_PATH.resolve()
    assert sys.modules["ib_async"].FlexReport is flexreport.FlexReport
