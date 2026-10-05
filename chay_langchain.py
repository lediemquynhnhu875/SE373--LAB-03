"""Real LangChain tools and three executable LangGraph workflows, without an LLM."""

from dataclasses import asdict
from datetime import datetime
from importlib.metadata import version
import json
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import TypedDict

from langchain.tools import tool
from langgraph.graph import END, START, StateGraph

from btvn03_flight_agent import (
    AgentHarness, AgentState, BaseFlightAgent, BookingRequest, Cabin,
    Flight, FlightTools, MOCK_FLIGHTS, RunTrace, ToolResult,
    print_table, scenario_set,
)


def request_from_dict(data: dict) -> BookingRequest:
    return BookingRequest(**{**data, "cabin": Cabin(data.get("cabin", "economy"))})


class LangChainFlightTools:
    """Adapt mock operations to actual StructuredTool.invoke calls."""

    def __init__(self):
        self.backend = FlightTools(MOCK_FLIGHTS)
        self.calls = []

        @tool
        def search_flights(request_data: dict) -> dict:
            """Search mock flights using a structured booking request."""
            result = self.backend.search_flights(request_from_dict(request_data))
            return {"ok": result.ok, "data": [asdict(f) for f in result.data], "error": result.error}

        @tool
        def hold_ticket(request_data: dict, flight_id: str) -> dict:
            """Hold a mock flight only with explicit consent and valid constraints."""
            request = request_from_dict(request_data)
            flight = next((f for f in self.backend.flights if f.flight_id == flight_id), None)
            if flight is None:
                return {"ok": False, "error": "Unknown flight", "data": None}
            harness = AgentHarness.default()
            errors = harness.data.validate_request(request) + harness.data.validate_selected_flight(request, flight)
            permission = harness.permission.can_book(request, flight, RunTrace("tool_guard"))
            if errors or not permission.ok:
                return {"ok": False, "error": "; ".join(errors) or permission.error, "data": None}
            result = self.backend.hold_ticket(request, flight)
            return {"ok": result.ok, "data": {"hold_code": result.data["hold_code"]} if result.ok else None, "error": result.error}

        @tool
        def issue_ticket(request_data: dict, hold_code: str) -> dict:
            """Issue a mock ticket after rechecking consent and the held flight."""
            request = request_from_dict(request_data)
            flight = self.backend.holds.get(hold_code)
            if flight is None:
                return {"ok": False, "error": "Unknown hold", "data": None}
            harness = AgentHarness.default()
            errors = harness.data.validate_request(request) + harness.data.validate_selected_flight(request, flight)
            permission = harness.permission.can_book(request, flight, RunTrace("tool_guard"))
            if errors or not permission.ok:
                return {"ok": False, "error": "; ".join(errors) or permission.error, "data": None}
            result = self.backend.issue_ticket(hold_code, request)
            return asdict(result)

        self.registry = {t.name: t for t in (search_flights, hold_ticket, issue_ticket)}

    def invoke(self, name, arguments):
        self.calls.append(name)
        result = self.registry[name].invoke(arguments)
        return ToolResult(**result)

    def search_flights(self, request):
        result = self.invoke("search_flights", {"request_data": asdict(request)})
        result.data = [Flight(**{**f, "cabin": Cabin(f["cabin"])}) for f in result.data]
        return result

    def hold_ticket(self, request, flight):
        return self.invoke("hold_ticket", {"request_data": asdict(request), "flight_id": flight.flight_id})

    def issue_ticket(self, hold_code, request):
        return self.invoke("issue_ticket", {"request_data": asdict(request), "hold_code": hold_code})


class GraphState(TypedDict):
    booking: AgentState


class LangGraphFlightAgent(BaseFlightAgent):
    def __init__(self, pattern):
        super().__init__(LangChainFlightTools(), AgentHarness.default())
        self.name = pattern
        graph = StateGraph(GraphState)
        actions = {
            "validate": self.validate,
            "plan": self.plan,
            "observe": self.observe,
            "search": self._search,
            "select": self._select_cheapest_valid,
            "evaluate": self.evaluate_candidate,
            "book": self._book_selected,
            "handoff": self.handoff,
        }
        active_nodes = {"validate", "search", "book", "handoff"}
        active_nodes.update({"observe", "select"} if pattern == "ReAct" else {"plan", "evaluate" if pattern == "Hybrid" else "select"})
        for name, action in actions.items():
            if name in active_nodes:
                graph.add_node(name, self.node(name, action))
        graph.add_edge(START, "validate")
        entry_node = "observe" if pattern == "ReAct" else "plan"
        graph.add_conditional_edges("validate", lambda s: "handoff" if s["booking"].errors else entry_node, ["handoff", entry_node])
        graph.add_edge("handoff", END)
        graph.add_edge("book", "handoff")
        if pattern == "ReAct":
            graph.add_conditional_edges("observe", self.next_action, ["search", "select", "book", "handoff"])
            graph.add_edge("search", "observe")
            graph.add_edge("select", "observe")
        else:
            graph.add_edge("plan", "search")
            next_node = "evaluate" if pattern == "Hybrid" else "select"
            graph.add_conditional_edges("search", lambda s: "handoff" if s["booking"].errors else next_node, ["handoff", next_node])
            if pattern == "Hybrid":
                graph.add_conditional_edges("evaluate", self.after_evaluate, ["evaluate", "book", "handoff"])
            else:
                graph.add_conditional_edges("select", lambda s: "book" if s["booking"].selected else "handoff", ["book", "handoff"])
        self.graph = graph.compile()

    @staticmethod
    def node(name, action):
        def execute(state):
            booking = state["booking"]
            booking.trace.add("LangGraph node: " + name)
            action(booking)
            return {"booking": booking}
        return execute

    def validate(self, state):
        errors = self.harness.data.validate_request(state.request)
        state.errors.extend(errors)
        state.trace.constraint_violations.extend(errors)

    def plan(self, state):
        state.trace.add("Plan: search -> select/validate -> permission -> book -> handoff")

    def observe(self, state):
        state.trace.add(f"Observation: candidates={len(state.candidates)}, selected={state.selected is not None}")

    @staticmethod
    def next_action(state):
        booking = state["booking"]
        if booking.errors:
            return "handoff"
        if not booking.candidates:
            return "search"
        return "select" if booking.selected is None else "book"

    def evaluate_candidate(self, state):
        if not state.candidates:
            state.errors.append("No valid candidate remains.")
            return
        flight = state.candidates.pop(0)
        errors = self.harness.data.validate_selected_flight(state.request, flight)
        if errors:
            state.trace.constraint_violations.extend(errors)
            state.trace.add(f"React: skip {flight.flight_id}: {'; '.join(errors)}")
        else:
            state.selected = flight

    @staticmethod
    def after_evaluate(state):
        booking = state["booking"]
        if booking.selected:
            return "book"
        return "handoff" if booking.errors else "evaluate"

    def handoff(self, state):
        state.trace.final_payload = self.harness.handoff.build(state)
        state.trace.completed, errors = self.harness.completion.is_complete(state)
        if not state.trace.completed:
            state.errors.extend(errors)
            state.trace.final_payload = self.harness.handoff.build(state)

    def run(self, request):
        started = perf_counter()
        self.tools.calls.clear()
        state = AgentState(request=request, trace=RunTrace(self.name))
        result = self.graph.invoke({"booking": state}, {"recursion_limit": 100})
        trace = result["booking"].trace
        trace.elapsed_ms = (perf_counter() - started) * 1000
        return trace


def main():
    metadata = {"time": datetime.now().astimezone().isoformat(), "langchain": version("langchain"), "langgraph": version("langgraph"), "controller": "deterministic; no LLM"}
    print("REAL LANGCHAIN TOOLS + COMPILED LANGGRAPH WORKFLOWS")
    print(json.dumps(metadata, ensure_ascii=False))
    rows, records = [], []
    output_dir = Path(__file__).resolve().parent
    for pattern in ("ReAct", "Plan-then-Execute", "Hybrid"):
        agent = LangGraphFlightAgent(pattern)
        (output_dir / f"graph_{pattern}.mmd").write_text(agent.graph.get_graph().draw_mermaid(), encoding="utf-8")
        traces = []
        for index, request in enumerate(scenario_set(), 1):
            trace = agent.run(request)
            traces.append(trace)
            payload = trace.final_payload
            expected = request.consent_to_book
            assert trace.completed == expected, (pattern, index, payload)
            assert trace.tool_calls == len(agent.tools.calls)
            if expected:
                assert payload["status"] == "ISSUED" and payload["ticket_code"]
                assert payload["total_price"] <= request.budget
            else:
                assert not payload["ticket_code"] and not payload["hold_code"]
                assert agent.tools.calls == ["search_flights"]
            print(f"{pattern} | case={index} | completed={trace.completed} | status={payload['status']} | calls={agent.tools.calls} | PASS")
            records.append({"scenario": index, "request": asdict(request), "trace": asdict(trace), "langchain_tool_calls": list(agent.tools.calls)})
        rows.append({"agent": pattern, "success_rate": mean(t.completed for t in traces), "avg_steps": mean(len(t.steps) for t in traces), "avg_tool_calls": mean(t.tool_calls for t in traces), "avg_permission_checks": mean(t.permission_checks for t in traces), "constraint_violations": sum(len(t.constraint_violations) for t in traces), "avg_elapsed_ms": mean(t.elapsed_ms for t in traces)})
    print("\nKET QUA LANGCHAIN/LANGGRAPH")
    print_table(rows)
    print("\nPASS: 18/18 expected outcomes; 15 mock tickets issued, 3 requests denied.")
    evidence = {"environment": metadata, "metrics": rows, "runs": records}
    (output_dir / "minh_chung_langchain.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
