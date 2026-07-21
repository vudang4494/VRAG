"""Sufficient Context Gate.

Evaluates whether the retrieved context contains enough information to answer
the user's query before spending time on full generation.
"""

from typing import Any

from loguru import logger

from api.routes._utils import llm_complete


async def check_sufficient_context(
    query: str, context: str, llm_model: str, timeout: float = 10.0
) -> dict[str, Any]:
    """Check if the context is sufficient to answer the query.

    Args:
        query: The user's original or rewritten question.
        context: The assembled (and optionally compressed) text from retrieved chunks.
        llm_model: The fast LLM model to use (e.g., settings.light_llm).
        timeout: Maximum time in seconds to wait for the LLM response.

    Returns:
        dict with keys:
            "is_sufficient": bool (True if yes, False if no)
            "reason": str (short explanation from the LLM)
    """
    if not context.strip():
        return {"is_sufficient": False, "reason": "Empty context"}

    prompt = f"""Bạn là một chuyên gia đánh giá thông tin.
Nhiệm vụ của bạn là kiểm tra xem ĐOẠN VĂN BẢN (Context) dưới đây có chứa đủ thông tin để trả lời CÂU HỎI (Query) hay không.

CÂU HỎI: {query}

ĐOẠN VĂN BẢN:
{context}

YÊU CẦU BẮT BUỘC:
- Trả lời bắt đầu bằng đúng một từ "YES" nếu CÓ đủ thông tin (hoặc có một phần lớn thông tin).
- Trả lời bắt đầu bằng đúng một từ "NO" nếu KHÔNG CÓ thông tin nào liên quan, hoặc thông tin hoàn toàn không giải quyết được câu hỏi.
- Sau chữ YES/NO, thêm dấu hai chấm ":" và giải thích ngắn gọn trong 1 câu tại sao.

Ví dụ:
YES: Đoạn văn bản có đề cập đến chi tiết X và Y mà câu hỏi yêu cầu.
NO: Đoạn văn bản chỉ nói về Z, hoàn toàn không có thông tin về vấn đề được hỏi.

TRẢ LỜI CỦA BẠN:"""

    try:
        # LLM call using the fast model. Temperature 0.0 for deterministic evaluation.
        response = await llm_complete(
            model=llm_model, prompt=prompt, max_tokens=60, temperature=0.0
        )
        response = response.strip()

        is_sufficient = True
        reason = response

        # Parse YES/NO
        upper_resp = response.upper()
        if upper_resp.startswith("NO") or " NO:" in upper_resp or " NO " in upper_resp[:10]:
            is_sufficient = False

        logger.debug(
            f"[SufficientContextGate] query='{query[:40]}...' -> sufficient={is_sufficient}, reason='{reason}'"
        )

        return {"is_sufficient": is_sufficient, "reason": reason}
    except Exception as e:
        logger.warning(
            f"[SufficientContextGate] LLM error: {e}. Defaulting to True to avoid blocking."
        )
        # Fail open: if the check fails (e.g., timeout), allow generation to proceed
        return {"is_sufficient": True, "reason": "Gate check failed, defaulting to True"}
