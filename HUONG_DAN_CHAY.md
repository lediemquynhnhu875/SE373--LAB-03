# Chạy LangChain/LangGraph và lấy minh chứng

## 1. Mở Terminal trong VS Code

Chạy từng lệnh trong thư mục bài tập (macOS, zsh):

```bash
cd /Users/nhule/Desktop/UIT-DS-SJ/SE373/BTVN03
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt --index-url https://pypi.org/simple
python -m pip check
```

Nếu `.venv` đã được tạo và cài thư viện, chỉ cần chạy `source .venv/bin/activate`.
Trong VS Code, chọn **Python: Select Interpreter** và chọn `.venv/bin/python`.

## 2. Chạy bản dùng thư viện thực tế

```bash
set -o pipefail
python chay_langchain.py | tee ket_qua_langchain.txt
```

`set -o pipefail` giúp Terminal trả mã lỗi nếu Python chạy lỗi dù có dùng `tee`.
Hoặc chạy trực tiếp, không cần kích hoạt môi trường:

```bash
.venv/bin/python chay_langchain.py
```

`chay_langchain.py` thực sự tạo tool bằng `from langchain.tools import tool`, gọi
`StructuredTool.invoke()`, tạo ba `StateGraph`, gọi `compile()` và `graph.invoke()`.
File `btvn03_flight_agent.py` chứa dữ liệu mock và các lớp harness dùng chung;
chạy riêng file đó vẫn là bản mô phỏng ban đầu.

Không cần API key, tài khoản LangSmith hoặc mô hình trả phí. Bộ điều khiển của
ba workflow được lập trình bằng quy tắc. Chưa có LLM tự suy luận/chọn tool:
ReAct ở đây là vòng điều khiển theo observation, Plan-then-Execute là kế hoạch
cố định, Hybrid có vòng bỏ qua candidate vi phạm. Nếu giảng viên yêu cầu agent
do LLM điều khiển, cần bổ sung mô hình và đánh giá riêng.

## 3. Minh chứng

Sau khi chạy thành công:

- `ket_qua_langchain.txt`: log Terminal, phiên bản thư viện, các tool thực sự
  được gọi trong từng kịch bản và bảng so sánh.
- `minh_chung_langchain.json`: phiên bản, thời điểm chạy, metrics, request,
  trace, handoff và danh sách tool đã gọi cho mỗi lượt.
- `graph_ReAct.mmd`, `graph_Plan-then-Execute.mmd`, `graph_Hybrid.mmd`: sơ đồ
  Mermaid xuất trực tiếp từ graph đã compile, có thể đưa vào báo cáo.

Chụp màn hình phần phiên bản thư viện, bảng so sánh, một lượt đặt vé thành công
và một lượt `completed=False` với `calls=['search_flights']` khi chưa có consent.

Kết quả mong đợi với dữ liệu hiện tại là 18/18 lượt kiểm tra đúng hành vi:
mỗi agent xuất 5 vé mock và từ chối 1 yêu cầu chưa được đồng ý. Tỷ lệ xuất vé
là 5/6 (83,33%); tỷ lệ đúng hành vi trên 6 kịch bản là 100%.
Mã vé/giữ chỗ và thời gian chạy thay đổi giữa các lần chạy.

`avg_steps` đếm dòng trace được ghi, không phải token suy luận hay số lần gọi
LLM. `avg_elapsed_ms` là thời gian workflow cục bộ với tool mock; không phản
ánh độ trễ mô hình hay API đặt vé. Sáu kịch bản này chưa đủ để kết luận mẫu
agent nào tốt hơn trong môi trường thực tế.

Các ngày bay mock nằm trong tháng 10-11/2026. Sau các ngày này, harness sẽ
từ chối ngày quá khứ: cần cập nhật đồng bộ ngày trong dữ liệu và kịch bản.

## 4. Lỗi thường gặp

- `ModuleNotFoundError: No module named 'langchain'`: dùng `.venv/bin/python`
  hoặc kích hoạt môi trường rồi cài `requirements.txt`.
- Không tải được thư viện: kiểm tra kết nối mạng và chạy lại lệnh cài có
  `--index-url https://pypi.org/simple`.
- Terminal in `LangChain available: False`: bạn đang chạy bản mô phỏng bằng
  Python khác; dùng `.venv/bin/python chay_langchain.py`.

## Tài liệu

- [LangChain tools](https://docs.langchain.com/oss/python/langchain/tools)
- [LangGraph Graph API](https://docs.langchain.com/oss/python/langgraph/graph-api)
