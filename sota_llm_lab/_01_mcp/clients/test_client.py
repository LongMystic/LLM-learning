import requests

BASE_URL = 'http://localhost:8081'

def call_remote_tool(tool: str, args: dict):
    resp = requests.post(
        f"{BASE_URL}/call_tool",
        json={
            'tool': tool,
            'args': args,
        }
    )
    resp.raise_for_status()
    data = resp.json()
    return data


if __name__ == "__main__":
    # 1 ) test echo
    print("=== echo ===")
    print(call_remote_tool("echo", {"text": "hello MCP"}))

    # 2 ) test read_file
    print("=== read_file ===")
    print(call_remote_tool("read_file", {"path": "../README.md"})["result"][:200])

    # 3 ) test search_files
    print("=== search_files ===")
    print(call_remote_tool("search_files", {"root": "..", "pattern": "*.py"}))

    # 4 ) test api call

    # 5 ) test grep_code
    print("=== grep_code ===")
    print(call_remote_tool("grep_code", {"path": "../README.md", "pattern": "MCP"}))
    # 6 ) test read_log_file
    print("=== read_log_file ===")
    print(call_remote_tool("read_log", {"path": "../README.md", "last_n_lines": 10}))
    # 7 ) test query_sqlite
    print("=== query_sqlite ===")
    print(call_remote_tool("query_sqlite", {"db_path": "C:\\LongVK\\Working\\sqlite\\longvk.db", "query": "SELECT * FROM test"}))