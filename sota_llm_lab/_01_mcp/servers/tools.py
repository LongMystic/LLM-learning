from pathlib import Path
from typing import Any, Dict, List
import os
import re
import requests
import sqlite3


def get_absolute_path(path: str) -> str:
    return os.path.abspath(path)


def echo(text: str) -> str:
    return text


def read_file(path: str, max_chars: int = 2000) -> str:
    print(f"Absolute path: {get_absolute_path(path)}")
    p = Path(path)
    if not p.exists():
        return f"[ERROR] File not found: {path}"
    if not p.is_file():
        return f"[ERROR] Not a file: {path}"
    content = p.read_text(encoding="utf-8", errors="ignore")
    if len(content) > max_chars:
        content = content[:max_chars] + "\n...[TRUNCATED]..."
    return content


def search_files(root: str, pattern: str = "*.py", max_results: int = 50) -> List[str]:
    base = Path(root)
    if not base.exists():
        return []
    return [str(p) for i, p in enumerate(base.rglob(pattern)) if i < max_results]


# ---------- Mục 3: Kết nối dịch vụ ngoài ----------
def call_http(
    url: str,
    method: str = "GET",
    params: Dict[str, Any] | None = None,
    max_chars: int = 2000,
) -> str:
    """Gọi REST API (GET/POST). params: query cho GET hoặc body JSON cho POST."""
    if requests is None:
        return "[ERROR] Cần cài: pip install requests"
    method = method.upper()
    try:
        if method == "GET":
            resp = requests.get(url, params=params or {}, timeout=15)
        elif method == "POST":
            resp = requests.post(url, json=params, timeout=15)
        else:
            return f"[ERROR] Method không hỗ trợ: {method}"
        resp.raise_for_status()
        text = resp.text
        if len(text) > max_chars:
            text = text[:max_chars] + "\n...[TRUNCATED]..."
        return text
    except Exception as e:
        return f"[ERROR] {e}"


# ---------- Mục 4: Mini-project toolbox ----------
def grep_code(
    path: str,
    pattern: str,
    max_lines: int = 50,
    file_pattern: str = "*.py",
) -> str:
    """Tìm dòng chứa pattern trong file hoặc thư mục. Trả về dạng path:line_no: content."""
    p = Path(path)
    if not p.exists():
        return f"[ERROR] Not found: {path}"
    results: List[str] = []
    try:
        regex = re.compile(pattern, re.IGNORECASE)
    except re.error:
        regex = re.compile(re.escape(pattern), re.IGNORECASE)
    files: List[Path] = [p] if p.is_file() else list(p.rglob(file_pattern))
    for f in files:
        if not f.is_file():
            continue
        try:
            for i, line in enumerate(f.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
                if regex.search(line):
                    results.append(f"{f}:{i}: {line.strip()}")
                    if len(results) >= max_lines:
                        return "\n".join(results)
        except Exception:
            continue
    return "\n".join(results) if results else "(không có kết quả)"


def read_log(path: str, last_n_lines: int = 100, max_chars: int = 4000) -> str:
    """Đọc N dòng cuối của file (phù hợp file log)."""
    p = Path(path)
    if not p.exists() or not p.is_file():
        return f"[ERROR] File not found: {path}"
    try:
        lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()
        tail = lines[-last_n_lines:] if len(lines) > last_n_lines else lines
        text = "\n".join(tail)
        if len(text) > max_chars:
            text = text[-max_chars:] + "\n...[TRUNCATED]..."
        return text
    except Exception as e:
        return f"[ERROR] {e}"


def query_sqlite(db_path: str, query: str, max_rows: int = 100) -> str:
    """Thực thi SELECT trên file SQLite. Chỉ cho phép câu SELECT."""
    if sqlite3 is None:
        return "[ERROR] Module sqlite3 không có."
    q = query.strip().upper()
    if not q.startswith("SELECT"):
        return "[ERROR] Chỉ cho phép câu SELECT."
    p = Path(db_path)
    if not p.exists() or not p.is_file():
        return f"[ERROR] File DB not found: {db_path}"
    try:
        conn = sqlite3.connect(str(p))
        conn.row_factory = sqlite3.Row
        cur = conn.execute(query, ())
        rows = cur.fetchmany(max_rows)
        conn.close()
        if not rows:
            return "(0 rows)"
        first = rows[0]
        lines = [" | ".join(first.keys())]
        for r in rows:
            lines.append(" | ".join(str(r[k]) for k in first.keys()))
        return "\n".join(lines)
    except Exception as e:
        return f"[ERROR] {e}"


# Simple dispatcher: tên tool -> hàm
TOOLS = {
    "echo": echo,
    "read_file": read_file,
    "search_files": search_files,
    "call_http": call_http,
    "grep_code": grep_code,
    "read_log": read_log,
    "query_sqlite": query_sqlite,
}


def call_tool(name: str, args: Dict[str, Any]) -> Any:
    if name not in TOOLS:
        raise ValueError(f"Unknown tool: {name}")
    func = TOOLS[name]
    return func(**args)

