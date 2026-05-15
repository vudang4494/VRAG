#!/usr/bin/env python3
"""Patch DRAFT_PROMPT in routes_v3.py"""

import re

path = "/Users/vudang/PythonLab/RAG/api/routes_v3.py"
content = open(path).read()

# Match from _DRAFT_PROMPT = """ to the closing """
pattern = r'(_DRAFT_PROMPT = """[\s\S]*?""")'
m = re.search(pattern, content)
if not m:
    print("Pattern not found")
    exit(1)

print("Found:", m.group(0)[:100])

# Build new prompt with actual newlines
new_lines = [
    '_DRAFT_PROMPT = """Bạn là trợ lý AI nội bộ. Trả lời câu hỏi sau bằng tiếng Việt,',
    "NGẮN GỌN (3-7 câu), dựa trên các đoạn văn bản tham khảo bên dưới.",
    "",
    "QUY TẮC (TUÂN THỦ TUYỆT ĐỐI):",
    "1. Nếu câu nào có thể trích dẫn được, thì kèm [chunk_id] ở cuối câu đó.",
    "   Không bắt buộc mọi câu đều phải có trích dẫn.",
    "2. CHỈ dùng tên thực thể (entity) XUẤT HIỆN trong các đoạn tham khảo.",
    "3. Bạn được phép tổng hợp, so sánh và diễn giải dựa trên ý nghĩa ngữ cảnh (semantic meaning).",
    "   Miễn là thông tin đưa ra không mâu thuẫn với sự thật trong nguồn và không bịa đặt sự thật mới (hallucination),",
    "   hãy đưa ra câu trả lời chi tiết và đầy đủ nhất thay vì từ chối.",
    "4. Nếu thật sự không có thông tin trong đoạn tham khảo, chỉ trả lời:",
    '   "Tôi không có đủ thông tin chắc chắn dựa trên tài liệu hiện có."',
    "",
    "Câu hỏi: {query}",
    "",
    "Dàn ý gợi ý (tham khảo):",
    "{outline}",
    "",
    "Các đoạn tham khảo:",
    "{context}",
    "",
    'Câu trả lời:"""',
]
new_prompt = "\n".join(new_lines)

new_content = content[: m.start()] + new_prompt + content[m.end() :]
open(path, "w").write(new_content)
print("Done! Replaced lines 49-70")
