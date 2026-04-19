from __future__ import annotations

from typing import Iterable

from .schemas import RetrievedChunk


class GroundedLLM:
    def __init__(
        self,
        provider: str,
        openai_api_key: str,
        openai_model: str,
        gemini_api_key: str,
        gemini_model: str,
    ) -> None:
        selected = provider.strip().lower()
        if selected not in {"local", "openai", "gemini"}:
            selected = "local"

        self.provider = selected
        self.openai_api_key = openai_api_key
        self.openai_model = openai_model
        self.gemini_api_key = gemini_api_key
        self.gemini_model = gemini_model

    def generate_answer(
        self,
        question: str,
        contexts: list[RetrievedChunk],
    ) -> tuple[str, str]:
        if not contexts:
            return (
                "Minh chua tim thay du lieu noi bo phu hop cho cau hoi nay. "
                "Hay bo sung tai lieu vao kho du lieu va thu lai.",
                "local",
            )

        prompt = self._build_prompt(question, contexts)

        if self.provider == "openai" and self.openai_api_key:
            try:
                return self._call_openai(prompt), "openai"
            except Exception:
                pass

        if self.provider == "gemini" and self.gemini_api_key:
            try:
                return self._call_gemini(prompt), "gemini"
            except Exception:
                pass

        return self._local_synthesis(question, contexts), "local"

    def _build_prompt(self, question: str, contexts: Iterable[RetrievedChunk]) -> str:
        context_blocks: list[str] = []
        for idx, item in enumerate(contexts, start=1):
            text = item.chunk.content.strip()
            if len(text) > 1800:
                text = f"{text[:1800]}..."
            context_blocks.append(
                "\n".join(
                    [
                        f"[Nguon {idx}]",
                        f"File: {item.chunk.source_name}",
                        f"Page: {item.chunk.page_number}",
                        f"Similarity: {item.similarity:.3f}",
                        f"Noi dung: {text}",
                    ]
                )
            )

        context_text = "\n\n".join(context_blocks)

        return "\n".join(
            [
                "Ban la tro ly hoc tap noi bo.",
                "Chi duoc su dung thong tin trong context de tra loi.",
                "Neu context khong du, phai noi ro khong du du lieu.",
                "Khi tra loi, neu co thong tin lien quan hay gan nhan [Nguon 1], [Nguon 2]...",
                "",
                "Context:",
                context_text,
                "",
                f"Cau hoi: {question}",
                "Tra loi ngan gon, ro rang, uu tien tieng Viet.",
            ]
        )

    def _call_openai(self, prompt: str) -> str:
        from openai import OpenAI

        client = OpenAI(api_key=self.openai_api_key)
        response = client.chat.completions.create(
            model=self.openai_model,
            temperature=0.1,
            messages=[
                {
                    "role": "system",
                    "content": "You are a grounded educational assistant."
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=700,
        )
        text = response.choices[0].message.content or ""
        return text.strip()

    def _call_gemini(self, prompt: str) -> str:
        import google.generativeai as genai

        genai.configure(api_key=self.gemini_api_key)
        model = genai.GenerativeModel(self.gemini_model)
        response = model.generate_content(prompt)

        text = getattr(response, "text", "") or ""
        if text.strip():
            return text.strip()

        candidates = getattr(response, "candidates", []) or []
        if candidates:
            parts = candidates[0].content.parts
            combined = "".join(getattr(part, "text", "") for part in parts)
            if combined.strip():
                return combined.strip()

        raise RuntimeError("Gemini returned an empty response")

    def _local_synthesis(self, question: str, contexts: list[RetrievedChunk]) -> str:
        if not contexts:
            return (
                "Khong co context de tong hop cau tra loi. "
                "Hay them tai lieu noi bo roi hoi lai."
            )

        lines = [
            "Tra loi duoc tong hop tu nguon noi bo da truy xuat:",
        ]

        for idx, item in enumerate(contexts, start=1):
            snippet = self._trim_for_summary(item.chunk.content)
            lines.append(
                f"{idx}. {snippet} [Nguon {idx}]"
            )

        lines.append(
            "Neu ban can do chinh xac cao hon, hay cau hinh OPENAI_API_KEY hoac GEMINI_API_KEY de bat che do LLM."
        )

        return "\n".join(lines)

    def _trim_for_summary(self, text: str, max_len: int = 280) -> str:
        flat = " ".join(text.split())
        if len(flat) <= max_len:
            return flat
        return f"{flat[:max_len].rstrip()}..."
