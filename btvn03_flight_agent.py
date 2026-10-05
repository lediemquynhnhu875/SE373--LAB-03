"""
BTVN#3 - Dung agent dat ve may bay bang LangChain/LangGraph style.

File nay co the chay truc tiep bang Python standard library:

    python3 btvn03_flight_agent.py

Y tuong nop bai:
- Tool mockup: search_flights, hold_ticket, issue_ticket.
- Harness: rang buoc du lieu, tieu chi hoan thanh kiem bang code,
  kiem quyen, ban giao.
- Agent voi 3 mau thiet ke: ReAct, Plan-then-Execute, Hybrid.
- Danh gia hieu qua 3 mau thiet ke tren cung tap kich ban.

Neu cai dat langchain/langgraph, cac tool/harness ben duoi co the duoc boc
lai bang @tool va StateGraph. Ban chay hien tai khong can API key/LLM.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from statistics import mean
from time import perf_counter
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
import re
import uuid


try:
    from langchain_core.tools import tool as langchain_tool  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    langchain_tool = None


try:
    from langgraph.graph import StateGraph  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    StateGraph = None


# ---------------------------------------------------------------------------
# Domain data
# ---------------------------------------------------------------------------


class Cabin(str, Enum):
    ECONOMY = "economy"
    BUSINESS = "business"


@dataclass(frozen=True)
class Flight:
    flight_id: str
    origin: str
    destination: str
    depart_date: str
    depart_time: str
    arrive_time: str
    airline: str
    price: int
    layovers: int
    cabin: Cabin = Cabin.ECONOMY


@dataclass
class BookingRequest:
    user_name: str
    origin: str
    destination: str
    depart_date: str
    passengers: int = 1
    budget: int = 5_000_000
    max_layovers: int = 1
    cabin: Cabin = Cabin.ECONOMY
    consent_to_book: bool = False


@dataclass
class ToolResult:
    ok: bool
    data: Any = None
    error: Optional[str] = None


@dataclass
class RunTrace:
    agent_name: str
    steps: List[str] = field(default_factory=list)
    tool_calls: int = 0
    permission_checks: int = 0
    constraint_violations: List[str] = field(default_factory=list)
    completed: bool = False
    elapsed_ms: float = 0.0
    final_payload: Optional[Dict[str, Any]] = None

    def add(self, message: str) -> None:
        self.steps.append(message)


@dataclass
class AgentState:
    request: BookingRequest
    candidates: List[Flight] = field(default_factory=list)
    selected: Optional[Flight] = None
    hold_code: Optional[str] = None
    ticket_code: Optional[str] = None
    errors: List[str] = field(default_factory=list)
    trace: Optional[RunTrace] = None


MOCK_FLIGHTS: List[Flight] = [
    Flight("VN100", "SGN", "HAN", "2026-10-10", "07:00", "09:10", "Vietnam Airlines", 2_350_000, 0),
    Flight("VJ142", "SGN", "HAN", "2026-10-10", "11:30", "13:45", "Vietjet Air", 1_450_000, 0),
    Flight("QH202", "SGN", "HAN", "2026-10-10", "17:00", "19:15", "Bamboo Airways", 1_800_000, 0),
    Flight("VN710", "SGN", "DAD", "2026-10-12", "09:05", "10:25", "Vietnam Airlines", 1_250_000, 0),
    Flight("VJ620", "SGN", "DAD", "2026-10-12", "14:20", "15:40", "Vietjet Air", 950_000, 0),
    Flight("VN401", "HAN", "PQC", "2026-10-15", "08:10", "12:20", "Vietnam Airlines", 3_700_000, 1),
    Flight("QH811", "HAN", "PQC", "2026-10-15", "15:00", "19:30", "Bamboo Airways", 3_200_000, 1),
    Flight("VN900", "SGN", "SIN", "2026-10-20", "09:00", "12:05", "Vietnam Airlines", 4_900_000, 0),
    Flight("TR305", "SGN", "SIN", "2026-10-20", "15:10", "18:15", "Scoot", 3_100_000, 0),
    Flight("VN950", "SGN", "NRT", "2026-11-05", "00:15", "08:00", "Vietnam Airlines", 9_800_000, 0),
    Flight("CX766", "SGN", "NRT", "2026-11-05", "11:25", "21:10", "Cathay Pacific", 7_400_000, 1),
]


# ---------------------------------------------------------------------------
# Mock tools
# ---------------------------------------------------------------------------


class FlightTools:
    """Mock tools. Trong LangChain co the boc cac method nay bang @tool."""

    def __init__(self, flights: Iterable[Flight]):
        self.flights = list(flights)
        self.holds: Dict[str, Flight] = {}

    def search_flights(self, request: BookingRequest) -> ToolResult:
        matches = [
            f
            for f in self.flights
            if f.origin == request.origin
            and f.destination == request.destination
            and f.depart_date == request.depart_date
            and f.cabin == request.cabin
        ]
        matches.sort(key=lambda f: (f.price, f.layovers, f.depart_time))
        if not matches:
            return ToolResult(False, [], "Khong tim thay chuyen bay phu hop tuyen/ngay.")
        return ToolResult(True, matches)

    def hold_ticket(self, request: BookingRequest, flight: Flight) -> ToolResult:
        total_price = flight.price * request.passengers
        if total_price > request.budget:
            return ToolResult(False, None, "Gia ve vuot ngan sach, khong duoc giu cho.")
        code = "HOLD-" + uuid.uuid4().hex[:8].upper()
        self.holds[code] = flight
        return ToolResult(True, {"hold_code": code, "flight": flight})

    def issue_ticket(self, hold_code: str, request: BookingRequest) -> ToolResult:
        flight = self.holds.get(hold_code)
        if flight is None:
            return ToolResult(False, None, "Ma giu cho khong hop le.")
        ticket_code = "TICKET-" + uuid.uuid4().hex[:10].upper()
        payload = {
            "ticket_code": ticket_code,
            "passenger": request.user_name,
            "flight_id": flight.flight_id,
            "route": f"{flight.origin}->{flight.destination}",
            "depart_date": flight.depart_date,
            "depart_time": flight.depart_time,
            "airline": flight.airline,
            "total_price": flight.price * request.passengers,
            "status": "ISSUED",
        }
        return ToolResult(True, payload)


def maybe_langchain_tools(tools: FlightTools) -> Dict[str, Callable[..., ToolResult]]:
    """
    Minh hoa cach map tool mockup sang LangChain.
    Ham nay van tra ve callable thuong neu langchain_core chua duoc cai.
    """

    registry: Dict[str, Callable[..., ToolResult]] = {
        "search_flights": tools.search_flights,
        "hold_ticket": tools.hold_ticket,
        "issue_ticket": tools.issue_ticket,
    }

    if langchain_tool is None:
        return registry

    # Optional path: may dung LangChain co the uncomment pattern nay de bind vao LLM.
    # De giu file chay on dinh trong moi truong cham, ta khong ep decorator len
    # method co kieu du lieu dataclass tuy bien.
    return registry


# ---------------------------------------------------------------------------
# Harness layers
# ---------------------------------------------------------------------------


class DataConstraintHarness:
    """Rang buoc la du lieu: request hop le va lua chon ve dung constraint."""

    airport_codes = {"SGN", "HAN", "DAD", "PQC", "SIN", "NRT"}

    def validate_request(self, request: BookingRequest) -> List[str]:
        errors: List[str] = []
        if not request.user_name.strip():
            errors.append("Thieu ten hanh khach.")
        if request.origin not in self.airport_codes:
            errors.append(f"San bay di khong hop le: {request.origin}.")
        if request.destination not in self.airport_codes:
            errors.append(f"San bay den khong hop le: {request.destination}.")
        if request.origin == request.destination:
            errors.append("San bay di va den khong duoc trung nhau.")
        if request.passengers < 1:
            errors.append("So hanh khach phai >= 1.")
        if request.budget <= 0:
            errors.append("Ngan sach phai > 0.")
        try:
            depart = datetime.strptime(request.depart_date, "%Y-%m-%d").date()
            if depart < date.today():
                errors.append("Ngay bay khong duoc nam trong qua khu.")
        except ValueError:
            errors.append("Ngay bay phai co dinh dang YYYY-MM-DD.")
        return errors

    def validate_selected_flight(self, request: BookingRequest, flight: Flight) -> List[str]:
        errors: List[str] = []
        total = flight.price * request.passengers
        if flight.origin != request.origin or flight.destination != request.destination:
            errors.append("Chuyen bay sai hanh trinh.")
        if flight.depart_date != request.depart_date:
            errors.append("Chuyen bay sai ngay.")
        if total > request.budget:
            errors.append(f"Gia {total:,} VND vuot ngan sach {request.budget:,} VND.")
        if flight.layovers > request.max_layovers:
            errors.append("So diem dung vuot gioi han.")
        return errors


class CompletionHarness:
    """Tieu chi hoan thanh kiem bang code."""

    def is_complete(self, state: AgentState) -> Tuple[bool, List[str]]:
        errors: List[str] = []
        if state.ticket_code is None:
            errors.append("Chua xuat ve.")
        if state.final_payload_status() != "ISSUED":
            errors.append("Trang thai ve chua phai ISSUED.")
        if state.selected is None:
            errors.append("Chua co chuyen bay duoc chon.")
        return len(errors) == 0, errors


def _state_final_payload_status(self: AgentState) -> Optional[str]:
    if self.trace and self.trace.final_payload:
        return self.trace.final_payload.get("status")
    return None


AgentState.final_payload_status = _state_final_payload_status  # type: ignore[attr-defined]


class PermissionHarness:
    """Kiem quyen: booking/issue ve can consent va khong vuot ngan sach."""

    def can_book(self, request: BookingRequest, flight: Flight, trace: RunTrace) -> ToolResult:
        trace.permission_checks += 1
        total = flight.price * request.passengers
        if not request.consent_to_book:
            return ToolResult(False, None, "Nguoi dung chua xac nhan cho phep dat ve.")
        if total > request.budget:
            return ToolResult(False, None, "Khong duoc dat ve vi vuot ngan sach.")
        return ToolResult(True, {"approved": True})


class HandoffHarness:
    """Ban giao ket qua ro rang cho nguoi dung/he thong tiep theo."""

    def build(self, state: AgentState) -> Dict[str, Any]:
        flight = state.selected
        if flight is None:
            return {"status": "FAILED", "errors": state.errors}
        return {
            "status": "ISSUED" if state.ticket_code else "NEEDS_ATTENTION",
            "ticket_code": state.ticket_code,
            "hold_code": state.hold_code,
            "passenger": state.request.user_name,
            "flight": {
                "flight_id": flight.flight_id,
                "airline": flight.airline,
                "route": f"{flight.origin}->{flight.destination}",
                "depart_date": flight.depart_date,
                "depart_time": flight.depart_time,
                "arrive_time": flight.arrive_time,
                "layovers": flight.layovers,
            },
            "total_price": flight.price * state.request.passengers,
            "errors": state.errors,
        }


@dataclass
class AgentHarness:
    data: DataConstraintHarness
    completion: CompletionHarness
    permission: PermissionHarness
    handoff: HandoffHarness

    @classmethod
    def default(cls) -> "AgentHarness":
        return cls(
            data=DataConstraintHarness(),
            completion=CompletionHarness(),
            permission=PermissionHarness(),
            handoff=HandoffHarness(),
        )


# ---------------------------------------------------------------------------
# Agent designs
# ---------------------------------------------------------------------------


class BaseFlightAgent:
    name = "base"

    def __init__(self, tools: FlightTools, harness: AgentHarness):
        self.tools = tools
        self.harness = harness

    def run(self, request: BookingRequest) -> RunTrace:
        start = perf_counter()
        trace = RunTrace(agent_name=self.name)
        state = AgentState(request=request, trace=trace)
        request_errors = self.harness.data.validate_request(request)
        if request_errors:
            state.errors.extend(request_errors)
            trace.constraint_violations.extend(request_errors)
            trace.final_payload = self.harness.handoff.build(state)
            trace.elapsed_ms = (perf_counter() - start) * 1000
            return trace

        self._run_impl(state)
        completed, completion_errors = self.harness.completion.is_complete(state)
        trace.completed = completed
        if not completed:
            state.errors.extend(completion_errors)
        trace.final_payload = self.harness.handoff.build(state)
        trace.elapsed_ms = (perf_counter() - start) * 1000
        return trace

    def _run_impl(self, state: AgentState) -> None:
        raise NotImplementedError

    def _search(self, state: AgentState) -> ToolResult:
        assert state.trace is not None
        state.trace.tool_calls += 1
        result = self.tools.search_flights(state.request)
        if result.ok:
            state.candidates = result.data
        else:
            state.errors.append(result.error or "Search failed")
        return result

    def _select_cheapest_valid(self, state: AgentState) -> Optional[Flight]:
        assert state.trace is not None
        for flight in state.candidates:
            violations = self.harness.data.validate_selected_flight(state.request, flight)
            if not violations:
                state.selected = flight
                return flight
            state.trace.constraint_violations.extend(violations)
        state.errors.append("Khong co chuyen bay nao thoa tat ca rang buoc.")
        return None

    def _book_selected(self, state: AgentState) -> None:
        assert state.trace is not None
        if state.selected is None:
            state.errors.append("Khong the dat ve vi chua chon chuyen bay.")
            return

        permission = self.harness.permission.can_book(state.request, state.selected, state.trace)
        if not permission.ok:
            state.errors.append(permission.error or "Permission denied")
            return

        state.trace.tool_calls += 1
        hold = self.tools.hold_ticket(state.request, state.selected)
        if not hold.ok:
            state.errors.append(hold.error or "Hold failed")
            return
        state.hold_code = hold.data["hold_code"]

        state.trace.tool_calls += 1
        issued = self.tools.issue_ticket(state.hold_code, state.request)
        if not issued.ok:
            state.errors.append(issued.error or "Issue failed")
            return
        state.ticket_code = issued.data["ticket_code"]
        state.trace.final_payload = issued.data


class ReActFlightAgent(BaseFlightAgent):
    """
    ReAct: lap Thought -> Action -> Observation.
    Mau nay linh hoat, nhung hay ton nhieu buoc hon vi vua suy nghi vua goi tool.
    """

    name = "ReAct"

    def _run_impl(self, state: AgentState) -> None:
        assert state.trace is not None
        for _ in range(6):
            if not state.candidates:
                state.trace.add("Thought: can tim chuyen bay. Action: search_flights.")
                result = self._search(state)
                state.trace.add(f"Observation: {len(result.data or [])} ket qua.")
                if not result.ok:
                    return
                continue

            if state.selected is None:
                state.trace.add("Thought: can chon ve re nhat thoa constraint.")
                selected = self._select_cheapest_valid(state)
                state.trace.add(f"Observation: selected={selected.flight_id if selected else None}.")
                if selected is None:
                    return
                continue

            if state.ticket_code is None:
                state.trace.add("Thought: can xin quyen va xuat ve. Action: hold_ticket -> issue_ticket.")
                self._book_selected(state)
                return


class PlanThenExecuteFlightAgent(BaseFlightAgent):
    """
    Plan-then-Execute: lap ke hoach truoc, sau do chay tuan tu.
    Mau nay de kiem soat nhung kem linh hoat neu buoc giua bi loi.
    """

    name = "Plan-then-Execute"

    def _run_impl(self, state: AgentState) -> None:
        assert state.trace is not None
        plan = ["search", "select", "permission_and_book", "handoff"]
        state.trace.add("Plan: " + " -> ".join(plan))

        result = self._search(state)
        if not result.ok:
            return

        selected = self._select_cheapest_valid(state)
        if selected is None:
            return

        self._book_selected(state)


class HybridFlightAgent(BaseFlightAgent):
    """
    Lai/Hybrid: co plan tong the nhu Plan-then-Execute, nhung moi observation
    duoc dung de dieu chinh chon phuong an nhu ReAct.
    """

    name = "Hybrid"

    def _run_impl(self, state: AgentState) -> None:
        assert state.trace is not None
        state.trace.add("Plan: search -> rank -> validate -> book; react neu vi pham constraint.")
        result = self._search(state)
        if not result.ok:
            return

        ranked = sorted(state.candidates, key=lambda f: (f.price, f.layovers, f.depart_time))
        for flight in ranked:
            state.trace.add(f"Evaluate: {flight.flight_id} price={flight.price} layovers={flight.layovers}.")
            violations = self.harness.data.validate_selected_flight(state.request, flight)
            if violations:
                state.trace.constraint_violations.extend(violations)
                state.trace.add("React: bo qua option vi " + "; ".join(violations))
                continue
            state.selected = flight
            break

        if state.selected is None:
            state.errors.append("Khong tim duoc ve hop le sau khi react voi cac vi pham.")
            return

        self._book_selected(state)


# ---------------------------------------------------------------------------
# Lightweight parser and evaluation
# ---------------------------------------------------------------------------


AIRPORT_ALIASES = {
    "tp hcm": "SGN",
    "sai gon": "SGN",
    "saigon": "SGN",
    "sgn": "SGN",
    "ha noi": "HAN",
    "hanoi": "HAN",
    "han": "HAN",
    "da nang": "DAD",
    "dad": "DAD",
    "phu quoc": "PQC",
    "pqc": "PQC",
    "singapore": "SIN",
    "sin": "SIN",
    "tokyo": "NRT",
    "narita": "NRT",
    "nrt": "NRT",
}


def parse_user_text(text: str, default_user: str = "Nguyen Van A") -> BookingRequest:
    """Parser don gian de bien cau tu nhien thanh BookingRequest."""

    lower = text.lower()
    airports = []
    for alias, code in AIRPORT_ALIASES.items():
        if alias in lower and code not in airports:
            airports.append(code)
    origin, destination = (airports + ["SGN", "HAN"])[:2]

    date_match = re.search(r"\d{4}-\d{2}-\d{2}", text)
    depart_date = date_match.group(0) if date_match else "2026-10-10"

    budget_match = re.search(r"(\d+(?:[.,]\d+)?)\s*(trieu|triệu|m|million)", lower)
    budget = 5_000_000
    if budget_match:
        budget = int(float(budget_match.group(1).replace(",", ".")) * 1_000_000)

    passengers_match = re.search(r"(\d+)\s*(nguoi|người|hanh khach|passenger)", lower)
    passengers = int(passengers_match.group(1)) if passengers_match else 1

    consent = any(word in lower for word in ["xac nhan", "xác nhận", "dong y", "đồng ý", "dat luon", "đặt luôn"])
    return BookingRequest(
        user_name=default_user,
        origin=origin,
        destination=destination,
        depart_date=depart_date,
        passengers=passengers,
        budget=budget,
        consent_to_book=consent,
    )


def build_agents() -> List[BaseFlightAgent]:
    harness = AgentHarness.default()
    tools = FlightTools(MOCK_FLIGHTS)
    maybe_langchain_tools(tools)
    return [
        ReActFlightAgent(tools, harness),
        PlanThenExecuteFlightAgent(tools, harness),
        HybridFlightAgent(tools, harness),
    ]


def scenario_set() -> List[BookingRequest]:
    return [
        BookingRequest("An", "SGN", "HAN", "2026-10-10", budget=2_000_000, consent_to_book=True),
        BookingRequest("Binh", "SGN", "DAD", "2026-10-12", budget=1_000_000, consent_to_book=True),
        BookingRequest("Chi", "HAN", "PQC", "2026-10-15", budget=3_300_000, max_layovers=1, consent_to_book=True),
        BookingRequest("Dung", "SGN", "SIN", "2026-10-20", budget=3_500_000, consent_to_book=True),
        BookingRequest("Ha", "SGN", "NRT", "2026-11-05", budget=8_000_000, max_layovers=1, consent_to_book=True),
        BookingRequest("Lan", "SGN", "HAN", "2026-10-10", budget=1_600_000, consent_to_book=False),
    ]


def evaluate_agents(agents: List[BaseFlightAgent], scenarios: List[BookingRequest]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for agent in agents:
        traces = [agent.run(s) for s in scenarios]
        success = sum(1 for t in traces if t.completed)
        rows.append(
            {
                "agent": agent.name,
                "success_rate": success / len(traces),
                "avg_steps": mean(len(t.steps) for t in traces),
                "avg_tool_calls": mean(t.tool_calls for t in traces),
                "avg_permission_checks": mean(t.permission_checks for t in traces),
                "constraint_violations": sum(len(t.constraint_violations) for t in traces),
                "avg_elapsed_ms": mean(t.elapsed_ms for t in traces),
            }
        )
    return rows


def print_table(rows: List[Dict[str, Any]]) -> None:
    headers = [
        "agent",
        "success_rate",
        "avg_steps",
        "avg_tool_calls",
        "avg_permission_checks",
        "constraint_violations",
        "avg_elapsed_ms",
    ]
    widths = {h: max(len(h), *(len(f"{row[h]:.2f}") if isinstance(row[h], float) else len(str(row[h])) for row in rows)) for h in headers}
    print(" | ".join(h.ljust(widths[h]) for h in headers))
    print("-+-".join("-" * widths[h] for h in headers))
    for row in rows:
        values = []
        for h in headers:
            value = row[h]
            values.append((f"{value:.2f}" if isinstance(value, float) else str(value)).ljust(widths[h]))
        print(" | ".join(values))


def demo_single_request() -> None:
    print("\nDEMO MOT REQUEST TU NHIEN")
    request = parse_user_text("Dat ve Sai Gon di Ha Noi ngay 2026-10-10 ngan sach 2 trieu, dong y dat luon")
    agent = HybridFlightAgent(FlightTools(MOCK_FLIGHTS), AgentHarness.default())
    trace = agent.run(request)
    print("Agent:", trace.agent_name)
    print("Completed:", trace.completed)
    print("Steps:")
    for step in trace.steps:
        print(" -", step)
    print("Handoff:", trace.final_payload)


def demo_langgraph_mapping() -> None:
    print("\nLANGGRAPH MAPPING")
    if StateGraph is None:
        print("LangGraph chua duoc cai. Mapping logic: state -> search -> select/validate -> permission -> issue -> handoff.")
    else:
        print("LangGraph co san. Co the tao StateGraph voi node search/select/book/handoff tu cac method trong file.")


def main() -> None:
    print("BTVN#3 - Flight Booking Agent")
    print("LangChain available:", langchain_tool is not None)
    print("LangGraph available:", StateGraph is not None)
    rows = evaluate_agents(build_agents(), scenario_set())
    print("\nKET QUA DANH GIA")
    print_table(rows)
    demo_single_request()
    demo_langgraph_mapping()


if __name__ == "__main__":
    main()
