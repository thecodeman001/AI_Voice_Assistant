import os
import time
from typing import Generator, Tuple
from groq import Groq


def approx_tokens(text: str) -> int:
    return max(1, int(len(text) / 4))


class LLMClient:
    def __init__(self, model: str | None = None, temperature: float = 0.4, max_tokens: int = 180):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = model or os.getenv("GROQ_LLM_MODEL", "penai/gpt-oss-20b")
        self.temperature = temperature
        self.max_tokens = max_tokens
        seed_str = os.getenv("SEED", "")
        try:
            self.seed = int(seed_str) if seed_str.strip() != "" else None
        except Exception:
            self.seed = None

    def stream_chat(self, messages: list) -> Tuple[Generator[str, None, None], float]:
        t0 = time.perf_counter()
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            seed=self.seed,
            stream=True,
        )
        first_latency = None

        def gen():
            nonlocal first_latency
            for chunk in stream:
                try:
                    delta = chunk.choices[0].delta.content or ""
                except Exception:
                    delta = ""
                if delta and first_latency is None:
                    first_latency = (time.perf_counter() - t0) * 1000
                if delta:
                    yield delta
        return gen(), (first_latency or (time.perf_counter() - t0) * 1000)

    def complete(self, messages: list) -> Tuple[str, float, dict | None]:
        t0 = time.perf_counter()
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            seed=self.seed,
            stream=False,
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        txt = resp.choices[0].message.content
        usage = getattr(resp, "usage", None)
        return txt, latency_ms, usage
