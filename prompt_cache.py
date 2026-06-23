from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PromptCacheCall:
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    input_tokens: int = 0
    latency_seconds: float = 0.0

    @property
    def saved_input_tokens(self) -> int:
        return round(self.cache_read_input_tokens * 0.9)


@dataclass
class PromptCacheMetrics:
    calls: list[PromptCacheCall] = field(default_factory=list)

    def record(self, usage: dict | None, latency_seconds: float) -> None:
        usage = usage or {}
        self.calls.append(
            PromptCacheCall(
                cache_creation_input_tokens=usage.get("cache_creation_input_tokens", 0) or 0,
                cache_read_input_tokens=usage.get("cache_read_input_tokens", 0) or 0,
                input_tokens=usage.get("input_tokens", 0) or 0,
                latency_seconds=latency_seconds,
            )
        )

    def summary(self) -> dict:
        cache_creation = sum(call.cache_creation_input_tokens for call in self.calls)
        cache_read = sum(call.cache_read_input_tokens for call in self.calls)
        input_tokens = sum(call.input_tokens for call in self.calls)
        saved_tokens = sum(call.saved_input_tokens for call in self.calls)
        total_latency = sum(call.latency_seconds for call in self.calls)

        miss_latencies = [
            call.latency_seconds
            for call in self.calls
            if call.cache_creation_input_tokens > 0 and call.cache_read_input_tokens == 0
        ]
        hit_latencies = [
            call.latency_seconds
            for call in self.calls
            if call.cache_read_input_tokens > 0
        ]
        latency_reduction_percent = None

        if miss_latencies and hit_latencies:
            avg_miss = sum(miss_latencies) / len(miss_latencies)
            avg_hit = sum(hit_latencies) / len(hit_latencies)
            latency_reduction_percent = max(0.0, (1 - avg_hit / avg_miss) * 100)

        return {
            "calls": len(self.calls),
            "cache_creation_input_tokens": cache_creation,
            "cache_read_input_tokens": cache_read,
            "fresh_input_tokens": input_tokens,
            "estimated_saved_input_tokens": saved_tokens,
            "total_latency_seconds": total_latency,
            "latency_reduction_percent": latency_reduction_percent,
        }

    def format_report(self) -> str:
        data = self.summary()
        latency_reduction = data["latency_reduction_percent"]

        if latency_reduction is None:
            latency_text = "n/a (este nevoie de cel putin un miss si un hit in acelasi proces)"
        else:
            latency_text = f"{latency_reduction:.1f}%"

        return "\n".join(
            [
                "Prompt cache metrics:",
                f"- calls: {data['calls']}",
                f"- cache_creation_input_tokens: {data['cache_creation_input_tokens']}",
                f"- cache_read_input_tokens: {data['cache_read_input_tokens']}",
                f"- fresh_input_tokens: {data['fresh_input_tokens']}",
                f"- estimated_saved_input_tokens: {data['estimated_saved_input_tokens']}",
                f"- total_latency_seconds: {data['total_latency_seconds']:.3f}",
                f"- latency_reduction: {latency_text}",
            ]
        )
