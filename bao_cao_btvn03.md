# BTVN#3 - Dung agent dat ve may bay bang LangChain/LangGraph

## 1. Muc tieu

Bai lam xay dung mot agent dat ve may bay voi tool mockup va cac lop harness bat buoc. File chinh la `btvn03_flight_agent.py`, co the chay truc tiep:

```bash
python3 btvn03_flight_agent.py
```

Ban standard library tren la ban mo phong. Ban su dung thu vien thuc te la
`chay_langchain.py`: tool duoc tao bang `langchain.tools.tool`, goi qua
`.invoke()`, va ba workflow chay bang `StateGraph.compile().invoke()`.
Xem `HUONG_DAN_CHAY.md` de cai moi truong va lay minh chung.
Bo dieu khien la quy tac lap trinh, chua co LLM tu suy luan/chon tool.

## 2. Tool mockup

Code cai dat 3 tool:

- `search_flights(request)`: tim chuyen bay theo san bay di, san bay den, ngay bay va hang ghe.
- `hold_ticket(request, flight)`: giu cho neu gia khong vuot ngan sach.
- `issue_ticket(hold_code, request)`: xuat ve tu ma giu cho.

Du lieu chuyen bay la mock data trong bien `MOCK_FLIGHTS`, giup viec cham bai khong can API that, API key hay ket noi mang.

## 3. Harness

Bai lam cai dat du 4 lop harness:

- `DataConstraintHarness`: kiem tra request hop le, san bay hop le, ngay bay hop le, ngan sach, so hanh khach, hanh trinh va constraint cua chuyen bay.
- `CompletionHarness`: tieu chi hoan thanh duoc kiem bang code, gom co `ticket_code`, flight duoc chon va trang thai `ISSUED`.
- `PermissionHarness`: kiem quyen truoc khi dat ve. Agent chi duoc giu cho/xuat ve khi nguoi dung da dong y va gia khong vuot ngan sach.
- `HandoffHarness`: dong goi ket qua ban giao gom ma ve, ma giu cho, hanh khach, hanh trinh, gio bay, hang bay, tong tien va loi neu co.

## 4. Ba mau thiet ke agent

### 4.1. ReAct

`ReActFlightAgent` lap theo chu trinh:

`Thought -> Action -> Observation`

Agent tim chuyen bay, quan sat ket qua, chon ve hop le, roi xin quyen va xuat ve. Mau nay linh hoat, de debug bang trace, nhung thuong ton nhieu step hon.

### 4.2. Plan-then-Execute

`PlanThenExecuteFlightAgent` tao plan truoc:

`search -> select -> permission_and_book -> handoff`

Sau do agent chay tuan tu theo plan. Mau nay ro rang, de kiem soat, nhung it linh hoat hon khi tool tra ve ket qua khong mong doi.

### 4.3. Lai/Hybrid

`HybridFlightAgent` ket hop hai cach tren. Agent co plan tong the, nhung sau moi observation se react voi constraint: neu mot chuyen bay vuot ngan sach hoac vuot so diem dung thi bo qua va thu option tiep theo.

## 5. Danh gia

Ham `evaluate_agents()` chay 3 agent tren cung mot tap kich ban trong `scenario_set()`. Cac metric:

- `success_rate`: ty le dat ve thanh cong.
- `avg_steps`: so buoc ly luan trung binh.
- `avg_tool_calls`: so lan goi tool trung binh.
- `avg_permission_checks`: so lan kiem quyen trung binh.
- `constraint_violations`: tong so vi pham constraint phat hien.
- `avg_elapsed_ms`: thoi gian chay trung binh.

Ket qua mau khi chay tren may hien tai:

```text
agent             | success_rate | avg_steps | avg_tool_calls | avg_permission_checks | constraint_violations | avg_elapsed_ms
------------------+--------------+-----------+----------------+-----------------------+-----------------------+---------------
ReAct             | 0.83         | 5         | 2.67           | 1                     | 0                     | ...
Plan-then-Execute | 0.83         | 1         | 2.67           | 1                     | 0                     | ...
Hybrid            | 0.83         | 2         | 2.67           | 1                     | 0                     | ...
```

Gia tri thoi gian co the thay doi theo may. Mot kich ban khong thanh cong la truong hop nguoi dung chua xac nhan quyen dat ve, dung voi yeu cau `PermissionHarness`.

## 6. Nhan xet

- ReAct phu hop khi bai toan can quan sat va dieu chinh tung buoc. Trace de doc nhung so step cao.
- Plan-then-Execute phu hop voi workflow nghiep vu on dinh nhu dat ve, vi cac buoc ro rang va de audit.
- Hybrid la lua chon can bang nhat: van co plan de kiem soat, nhung co kha nang react khi flight candidate vi pham constraint.

Trong bai toan dat ve may bay, mau Hybrid nen duoc uu tien vi booking la tac vu co rui ro ve tien/quyen, can harness chat che, nhung van can linh hoat khi gia ve, diem dung, hoac option thay the thay doi.

## 7. Cach mo rong sang LangChain/LangGraph

- LangChain: boc `FlightTools.search_flights`, `hold_ticket`, `issue_ticket` thanh tool bang `@tool`, sau do bind vao LLM.
- LangGraph: tao `StateGraph(AgentState)` voi cac node `validate`, `search`, `select`, `permission`, `book`, `handoff`; canh dieu kien dung dua vao `CompletionHarness.is_complete()`.
- Khi dung API that, giu nguyen `PermissionHarness` de ngan agent tu y thanh toan/dat ve khi chua co xac nhan cua nguoi dung.

## 8. Ban LangChain/LangGraph da trien khai

Chay `.venv/bin/python chay_langchain.py` sau khi cai `requirements.txt`.
Ban nay tai su dung cac lop harness va tool mock trong file chinh, boc tool
thanh LangChain StructuredTool, va dieu phoi bang ba graph rieng:

- ReAct: `observe` chon `search`, `select` hoac `book` dua tren state, roi
  quay lai quan sat sau search/select.
- Plan-then-Execute: `plan -> search -> select -> book -> handoff`.
- Hybrid: `plan -> search -> evaluate`; neu candidate vi pham thi lap
  `evaluate` voi candidate tiep theo, neu hop le thi `book -> handoff`.

Moi graph bat dau bang validate va ket thuc bang handoff. Tool giu cho va
xuat ve cung kiem consent va constraint tai bien tool. Khong co LLM/API key.
Do do day la danh gia workflow xac dinh dung framework, khong phai danh gia
chat luong suy luan cua LLM.

Ket qua thuc te duoc luu trong `ket_qua_langchain.txt` va
`minh_chung_langchain.json`; so do graph trong cac file `graph_*.mmd`.
Khong dung bang ket qua ban standard library o muc 5 lam minh chung cho
ban LangChain. `avg_steps` dem dong trace, khong do token hay so lan goi LLM.
Voi 6 kich ban, ky vong moi agent xuat 5 ve va tu choi 1 yeu cau chua consent:
ty le xuat ve 83,33%, ty le dung hanh vi 100%. Can tap kich ban lon hon,
loi tool va bien dong gia de so sanh cac mau thiet ke mot cach thuyet phuc.
