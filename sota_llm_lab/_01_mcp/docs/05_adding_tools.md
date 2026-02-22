# 05 – Hướng dẫn thêm tool mới

Áp dụng cho **mục 3 (Kết nối dịch vụ ngoài)** và **mục 4 (Mini-project)** trong README.

---

## Quy trình chung khi thêm 1 tool

1. **Định nghĩa hàm** trong `servers/tools.py`: tên hàm, tham số, giá trị trả về (string hoặc JSON-serializable).
2. **Đăng ký** vào dict `TOOLS` trong cùng file.
3. **Cập nhật mô tả tool** trong `servers/ollama_tool_planner.py` (biến `tools_description`) để Ollama biết khi nào gọi tool và args là gì.
4. (Tùy chọn) Thêm test trong `clients/test_client.py`.

---

## Mục 3 – Kết nối tới dịch vụ ngoài

### Tool: `call_http`

- **Mục đích**: Gọi REST API public (GET hoặc POST) để lấy dữ liệu (thời tiết, giá crypto, …).
- **Tham số gợi ý**:
  - `url` (str): URL đầy đủ.
  - `method` (str): `"GET"` hoặc `"POST"`, mặc định `"GET"`.
  - `params` (dict, tùy chọn): query string (GET) hoặc body JSON (POST).
- **Trả về**: Nội dung response dạng text (hoặc JSON string). Nên giới hạn độ dài (vd. 2000 ký tự) để không làm tràn context.
- **Lưu ý bảo mật**: Chỉ cho phép URL thuộc whitelist (vd. `https://api.openweathermap.org`, `https://api.coingecko.com`) hoặc chỉ GET; tránh gọi URL nội bộ (localhost, 192.168.x) nếu không cần.

*(Sau khi viết xong hàm, đăng ký vào `TOOLS` và thêm mô tả vào `tools_description` trong `ollama_tool_planner.py`.)*

---

## Mục 4 – Mini-project (MCP toolbox)

### Tool: `grep_code`

- **Mục đích**: Tìm dòng chứa pattern trong file (hoặc trong nhiều file), tương tự grep.
- **Tham số gợi ý**:
  - `path` (str): file đơn hoặc thư mục (nếu thư mục thì quét file `.py`, `.md`, …).
  - `pattern` (str): chuỗi hoặc regex đơn giản để tìm.
  - `max_lines` (int): số dòng kết quả tối đa trả về (vd. 50).
- **Trả về**: Danh sách chuỗi dạng `"path:line_no: nội dung dòng"` hoặc 1 string gộp.

*(Đăng ký vào `TOOLS`, cập nhật prompt planner.)*

---

### Tool: `read_log`

- **Mục đích**: Đọc file log, ưu tiên “đuôi” file (dòng mới nhất).
- **Tham số gợi ý**:
  - `path` (str): đường dẫn file log.
  - `last_n_lines` (int): lấy N dòng cuối (mặc định 100).
  - `max_chars` (int): giới hạn tổng ký tự trả về.
- **Trả về**: Chuỗi nội dung N dòng cuối.

*(Đăng ký vào `TOOLS`, cập nhật prompt planner.)*

---

### Tool: `query_sqlite`

- **Mục đích**: Thực thi câu lệnh đọc dữ liệu (SELECT) trên file SQLite local.
- **Tham số gợi ý**:
  - `db_path` (str): đường dẫn file `.db` / `.sqlite`.
  - `query` (str): câu SQL (chỉ cho phép SELECT; từ chối DROP/INSERT/UPDATE/DELETE).
- **Trả về**: Kết quả dạng list of dict hoặc bảng dạng text; giới hạn số dòng (vd. 100).
- **Lưu ý bảo mật**: Validate `query` chỉ chứa SELECT; không cho path ra ngoài thư mục cho phép (vd. chỉ cho phép 1 thư mục data/).

*(Đăng ký vào `TOOLS`, cập nhật prompt planner.)*

---

## Đoạn mô tả tool dán vào `ollama_tool_planner.py`

Thêm vào biến `tools_description` (trong `plan_with_ollama`) các block sau:

```text
4) call_http
- Mô tả: Gọi REST API (GET/POST). Dùng khi user hỏi thời tiết, giá crypto, tin tức từ API.
- Cách gọi: {"tool": "call_http", "args": {"url": "https://...", "method": "GET", "params": {}}}

5) grep_code
- Mô tả: Tìm dòng chứa pattern trong file/thư mục code.
- Cách gọi: {"tool": "grep_code", "args": {"path": "đường/dẫn/file hoặc thư mục", "pattern": "chuỗi tìm", "max_lines": 50}}

6) read_log
- Mô tả: Đọc N dòng cuối của file log.
- Cách gọi: {"tool": "read_log", "args": {"path": "đường/dẫn/file.log", "last_n_lines": 100}}

7) query_sqlite
- Mô tả: Chạy câu SELECT trên file SQLite (.db).
- Cách gọi: {"tool": "query_sqlite", "args": {"db_path": "đường/dẫn/file.db", "query": "SELECT ... FROM ..."}}
```

---

## Sau khi thêm tool

- Chạy lại MCP server và test bằng `test_client.py` (gọi trực tiếp tool mới).
- Chạy `ollama_tool_planner.py` và thử vài câu user tự nhiên để kiểm tra Ollama có chọn đúng tool và args.
