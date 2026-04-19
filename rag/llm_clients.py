from __future__ import annotations

from typing import Iterable

from .ollama_http import generate_text
from .schemas import RetrievedChunk


class GroundedLLM:
    def __init__(
        self,
        provider: str,
        ollama_host: str,
        ollama_llm_model: str,
        ollama_fallback_model: str,
        ollama_num_ctx: int,
        ollama_temperature: float,
        ollama_timeout: float,
        openai_api_key: str,
        openai_model: str,
        gemini_api_key: str,
        gemini_model: str,
    ) -> None:
        selected = provider.strip().lower()
        if selected not in {"local", "ollama", "openai", "gemini"}:
            selected = "ollama"

        self.provider = selected
        self.ollama_host = ollama_host
        self.ollama_llm_model = ollama_llm_model
        fallback = (ollama_fallback_model or "").strip()
        self.ollama_fallback_model = fallback if fallback != ollama_llm_model else ""
        self.ollama_num_ctx = max(1024, int(ollama_num_ctx))
        self.ollama_temperature = min(1.0, max(0.0, float(ollama_temperature)))
        self.ollama_timeout = max(10.0, float(ollama_timeout))
        self.openai_api_key = openai_api_key
        self.openai_model = openai_model
        self.gemini_api_key = gemini_api_key
        self.gemini_model = gemini_model

    def generate_answer(
        self,
        question: str,
        contexts: list[RetrievedChunk],
        *,
        intent: str,
        confidence: float,
        evidence_sentences: list[str] | None = None,
    ) -> tuple[str, str]:
        if not contexts:
            return (
                "Minh chua tim thay du lieu noi bo phu hop cho cau hoi nay. "
                "Hay bo sung tai lieu vao kho du lieu va thu lai.",
                "local",
            )

        context_text = self._render_context_blocks(contexts)
        facts_prompt = self._build_facts_prompt(
            question=question,
            intent=intent,
            confidence=confidence,
            context_text=context_text,
            evidence_sentences=evidence_sentences or [],
        )

        facts_text, facts_provider = self._call_provider_chain(facts_prompt)
        if not facts_text:
            facts_text = self._local_extract_facts(contexts)
            facts_provider = "local"

        answer_prompt = self._build_answer_prompt(
            question=question,
            intent=intent,
            confidence=confidence,
            facts_text=facts_text,
        )

        answer_text, answer_provider = self._call_provider_chain(answer_prompt)
        if answer_text:
            return answer_text, answer_provider

        return self._local_synthesis(question, contexts, facts_text), facts_provider

    def _render_context_blocks(self, contexts: Iterable[RetrievedChunk]) -> str:
        blocks: list[str] = []
        for idx, item in enumerate(contexts, start=1):
            parent_text = (item.chunk.parent_content or item.chunk.content).strip()
            if len(parent_text) > 950:
                parent_text = f"{parent_text[:950]}..."

            child_text = item.chunk.content.strip()
            if len(child_text) > 420:
                child_text = f"{child_text[:420]}..."

            doc_type = (item.chunk.doc_type or "document").strip().lower()
            location_label = "Slide" if doc_type == "slide" else "Trang"
            location_number = item.chunk.parent_page_number or item.chunk.page_number

            blocks.append(
                "\n".join(
                    [
                        f"[Nguon {idx}]",
                        f"File: {item.chunk.source_name}",
                        f"Tieu de: {item.chunk.title or item.chunk.source_name}",
                        f"Mon/Lop/Chuong: {item.chunk.subject or '-'} / {item.chunk.grade or '-'} / {item.chunk.chapter or '-'}",
                        f"{location_label}: {location_number}",
                        f"Section: {item.chunk.section_path or item.chunk.parent_title or 'N/A'}",
                        f"Diem Dense/Sparse/RRF/Rerank: {item.dense_score:.3f} / {item.sparse_score:.3f} / {item.rrf_score:.3f} / {item.rerank_score:.3f}",
                        f"Doan child: {child_text}",
                        f"Noi dung parent: {parent_text}",
                    ]
                )
            )

        return "\n\n".join(blocks)

    def _build_facts_prompt(
        self,
        *,
        question: str,
        intent: str,
        confidence: float,
        context_text: str,
        evidence_sentences: list[str],
    ) -> str:
        evidence_block = "\n".join(f"- {row}" for row in evidence_sentences)
        if not evidence_block:
            evidence_block = "- Khong co evidence rut gon; hay trich truc tiep tu context."

        return "\n".join(
            [
                "Ban la bo loc su kien cho tro ly hoc tap noi bo.",
                "NHIEM VU PASS 1: Rut su kien dung va du tu context, khong duoc suy doan them.",
                "LUAT CUNG:",
                "1) CHI dung thong tin trong context/evidence.",
                "2) Moi su kien phai co trich dan [Nguon X].",
                "3) Neu thay mau thuan giua cac nguon, ghi ro MAU_THUAN + trich dan.",
                "4) Neu thieu thong tin de ket luan, ghi THIEU_DU_LIEU.",
                f"Loai cau hoi: {intent}",
                f"Do tin cay truy hoi: {confidence:.3f}",
                "",
                "Evidence rut gon:",
                evidence_block,
                "",
                "Context day du:",
                context_text,
                "",
                f"Cau hoi: {question}",
                "",
                "Dinh dang output bat buoc:",
                "FACTS:",
                "- <fact 1> [Nguon X]",
                "- <fact 2> [Nguon Y]",
                "CONFLICTS:",
                "- <neu co>",
                "MISSING:",
                "- <neu co>",
            ]
        )

    def _build_answer_prompt(
        self,
        *,
        question: str,
        intent: str,
        confidence: float,
        facts_text: str,
    ) -> str:
        return "\n".join(
            [
                "Ban la tro ly hoc tap noi bo.",
                "NHIEM VU PASS 2: Tong hop cau tra loi cuoi cung CHI tu FACTS ben duoi.",
                "LUAT CUNG:",
                "1) Khong them kien thuc ngoai FACTS.",
                "2) Moi y quan trong phai kem [Nguon X].",
                "3) Neu MISSING khong rong thi phai noi ro khong du du lieu.",
                "4) Neu CONFLICTS khong rong thi neu mau thuan, khong tu chon bua.",
                f"Loai cau hoi: {intent}",
                f"Do tin cay truy hoi: {confidence:.3f}",
                "",
                "FACT PACKAGE:",
                facts_text,
                "",
                f"Cau hoi: {question}",
                "",
                "Dinh dang output:",
                "1) Tra loi chinh (ngan gon, ro rang)",
                "2) Giai thich bo sung (neu can)",
                "3) Neu thieu du lieu: dua ra 1-2 cau hoi lam ro",
            ]
        )

    def _call_provider_chain(self, prompt: str) -> tuple[str, str]:
        if self.provider == "ollama" and self.ollama_llm_model:
            try:
                text, used_model = self._call_ollama_with_fallback(prompt)
                return text, f"ollama:{used_model}"
            except Exception:
                pass

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

        if self.ollama_llm_model:
            try:
                text, used_model = self._call_ollama_with_fallback(prompt)
                return text, f"ollama:{used_model}"
            except Exception:
                pass

        return "", "local"

    def _call_ollama(self, prompt: str, model: str) -> str:
        return generate_text(
            host=self.ollama_host,
            model=model,
            prompt=prompt,
            num_ctx=self.ollama_num_ctx,
            temperature=self.ollama_temperature,
            timeout=self.ollama_timeout,
        )

    def _call_ollama_with_fallback(self, prompt: str) -> tuple[str, str]:
        models: list[str] = []
        if self.ollama_llm_model:
            models.append(self.ollama_llm_model)
        if self.ollama_fallback_model and self.ollama_fallback_model not in models:
            models.append(self.ollama_fallback_model)

        if not models:
            raise RuntimeError("No Ollama model configured")

        last_error: Exception | None = None
        for index, model in enumerate(models):
            try:
                return self._call_ollama(prompt, model), model
            except Exception as exc:
                last_error = exc
                message = str(exc).casefold()
                if index < len(models) - 1 and self._should_retry_with_fallback(message):
                    continue
                if index < len(models) - 1:
                    continue

        if last_error is not None:
            raise last_error
        raise RuntimeError("Ollama generation failed")

    def _should_retry_with_fallback(self, message: str) -> bool:
        fallback_keywords = (
            "out of memory",
            "cuda",
            "gpu",
            "timeout",
            "timed out",
            "context length",
            "context window",
            "model requires more system memory",
        )
        return any(keyword in message for keyword in fallback_keywords)

    def _call_openai(self, prompt: str) -> str:
        from openai import OpenAI

        client = OpenAI(api_key=self.openai_api_key)
        response = client.chat.completions.create(
            model=self.openai_model,
            temperature=0.0,
            messages=[
                {
                    "role": "system",
                    "content": "You are a grounded educational assistant.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=900,
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

    def _local_extract_facts(self, contexts: list[RetrievedChunk]) -> str:
        lines = ["FACTS:"]
        for idx, item in enumerate(contexts, start=1):
            snippet = self._trim_for_summary(item.chunk.parent_content or item.chunk.content, max_len=240)
            lines.append(
                f"- {snippet} [Nguon {idx}]"
            )
        lines.append("CONFLICTS:")
        lines.append("- Chua xac dinh mau thuan tu local fallback.")
        lines.append("MISSING:")
        lines.append("- Co the thieu du lieu neu can lap luan sau hon.")
        return "\n".join(lines)

    def _local_synthesis(
        self,
        question: str,
        contexts: list[RetrievedChunk],
        facts_text: str,
    ) -> str:
        lines = [
            "Khong goi duoc model LLM nen minh tong hop bang local fallback.",
            "",
            "Tom tat su kien:",
            facts_text,
            "",
            f"Cau hoi goc: {question}",
            "Neu ban can cau tra loi chat luong cao hon, hay bat Ollama/OpenAI/Gemini.",
        ]

        if contexts:
            lines.append("")
            lines.append("Nguon da dung:")
            for idx, item in enumerate(contexts, start=1):
                location = item.chunk.parent_page_number or item.chunk.page_number
                lines.append(
                    f"- [Nguon {idx}] {item.chunk.source_name} (trang {location})"
                )

        return "\n".join(lines)

    def _trim_for_summary(self, text: str, max_len: int = 280) -> str:
        flat = " ".join(text.split())
        if len(flat) <= max_len:
            return flat
        return f"{flat[:max_len].rstrip()}..."
