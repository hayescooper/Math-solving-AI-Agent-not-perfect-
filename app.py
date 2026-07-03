import os
import gradio as gr
import requests
import PyPDF2

# ---------------------------------------------------------
# CẤU HÌNH OPENROUTER
# ---------------------------------------------------------
OPENROUTER_API_KEY = "YOUR-API-KEY-HERE"
SITE_URL = "https://huggingface.co/spaces/AugustinHuang/Math-solving"
APP_NAME = "Math-solving"

if not OPENROUTER_API_KEY:
    raise ValueError("Thiếu OPENROUTER_API_KEY. Hãy thêm vào Secrets của Hugging Face Space.")

# ---------------------------------------------------------
# SYSTEM PROMPT
# ---------------------------------------------------------
SYSTEM_PROMPT = """
Bạn là "Thầy Giáo Toán" đến từ Đại học Quốc gia TP.HCM (VNU-HCM). 
Phong cách của bạn:
1. Chuyên nghiệp, uyên bác nhưng gần gũi, tận tâm.
2. Luôn giải thích cặn kẽ, từng bước một (step-by-step) để sinh viên hiểu bản chất vấn đề, không chỉ đưa ra đáp số.
3. Sử dụng ngôn ngữ tiếng Việt chuẩn mực, rõ ràng.
4. Khi gặp công thức toán học, hãy trình bày đẹp mắt hoặc giải thích bằng lời văn dễ hiểu.
5. Nếu đề bài thiếu thông tin, hãy hỏi lại người học một cách khéo léo.
6. Mục tiêu tối thượng: Giúp người học tư duy logic và yêu thích môn toán.
7. Hãy cố gắng tạo 1 "mỏ neo" ở các dạng bài khó, ví dụ: giải xong một bài toán thực tế nếu quá dài dòng thì hãy tóm gọn lại ở phần cuối theo các bước cụ thể để giúp học sinh ghi nhớ tốt hơn
8. Hãy cố gắng giải 1 cách ngắn gọn nhất có thể vì bạn không có đủ token để nhiều lời
9. Viết công thức đơn giản, dễ đọc, KHÔNG dùng LaTeX phức tạp
10. Ví dụ format:
   - f'(x) = 0.025 - 0.00002x
   - x = (-2 + √(4 + 4×9×2100)) / 18 ≈ 15.16
   - Kết luận: x ≈ 15.16 (triệu đồng)
11. Sử dụng LaTeX cho công thức toán, bọc trong $$...$$ cho công thức riêng dòng, hoặc $...$ cho công thức inline.
   Ví dụ: $$x = \\frac{-b + \\sqrt{b^2 - 4ac}}{2a}$$
Hãy bắt đầu bằng lời chào ngắn gọn và sẵn sàng hỗ trợ.
"""

def extract_text_from_pdf(pdf_file):
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text.strip()
    except Exception as e:
        return f"Lỗi đọc PDF: {str(e)}"

def solve_math_problem_openrouter(user_input, pdf_file, image_file):
    context_text = user_input
    
    if pdf_file is not None:
        pdf_content = extract_text_from_pdf(pdf_file)
        if pdf_content:
            context_text += f"\n\n[Nội dung từ PDF]:\n{pdf_content}"
            
    if image_file is not None:
        context_text += "\n\n[Lưu ý: Hệ thống chưa hỗ trợ đọc ảnh trực tiếp qua API miễn phí này. Vui lòng gõ lại đề bài từ ảnh]."

    if not context_text.strip():
        yield "Em vui lòng nhập câu hỏi!"
        return

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": context_text}
    ]

    try:
        # Gọi API OpenRouter
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": SITE_URL, 
                "X-Title": APP_NAME,
            },
            json={
                "model": "nvidia/nemotron-3-nano-30b-a3b:free", # Model miễn phí
                "messages": messages,
                "temperature": 0.2,
                "stream": True, # Bật stream để hiển thị dần
            },
            stream=True
        )

        response.raise_for_status()
        
        full_response = ""
        for line in response.iter_lines():
            if line:
                # Xử lý dòng dữ liệu stream từ OpenRouter
                decoded_line = line.decode('utf-8')
                if decoded_line.startswith("data: "):
                    data_str = decoded_line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        import json
                        data = json.loads(data_str)
                        content = data['choices'][0]['delta'].get('content', '')
                        if content:
                            full_response += content
                            yield full_response
                    except:
                        continue

    except Exception as e:
        yield f"Lỗi kết nối: {str(e)}. Kiểm tra lại API Key hoặc thử lại sau."

# ---------------------------------------------------------
# GIAO DIỆN GRADIO (Giữ nguyên như cũ)
# ---------------------------------------------------------
with gr.Blocks(title="Thầy Giáo Toán VNU-HCM", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎓 Thầy Giáo Toán - ĐH Quốc gia TP.HCM")
    gr.Markdown("Sử dụng nvidia/nemotron-3-nano-30b-a3b qua OpenRouter API.")

    with gr.Row():
        with gr.Column(scale=1):
            user_input = gr.Textbox(label="Câu hỏi:", lines=5)
            pdf_input = gr.File(label="File PDF", file_types=[".pdf"])
            image_input = gr.Image(label="Ảnh đề bài", type="filepath")
            submit_btn = gr.Button("Gửi bài", variant="primary")
        with gr.Column(scale=2):
            output_box = gr.Markdown(label="Lời giải:")

    submit_btn.click(
        fn=solve_math_problem_openrouter,
        inputs=[user_input, pdf_input, image_input],
        outputs=output_box
    )

if __name__ == "__main__":
    demo.launch()