#!/usr/bin/env python3
"""
Create sample math documents for testing Math RAG
Generates markdown files with LaTeX formulas (simulating Nougat output)
"""

import os
from pathlib import Path

def create_sample_math_docs():
    """Create sample math documents in ocr_input/"""
    
    docs = {
        "sample_quadratic_functions.mmd": """# Chương 1: Hàm Số Bậc Hai

## 1.1 Định Nghĩa

**Hàm số bậc hai** là hàm số được xác định bởi công thức:

$$f(x) = ax^2 + bx + c$$

trong đó $a$, $b$, $c$ là các hệ số thực và $a \\neq 0$.

**Lưu ý:**
- Nếu $a = 0$, hàm số trở thành hàm bậc nhất: $f(x) = bx + c$
- Nếu $b = c = 0$, hàm số là $f(x) = ax^2$

## 1.2 Tính Chất Của Đồ Thị

Đồ thị của hàm số bậc hai $f(x) = ax^2 + bx + c$ là một **parabol** có:

- **Đỉnh parabol**: $I\\left(-\\frac{b}{2a}; -\\frac{\\Delta}{4a}\\right)$

  trong đó $\\Delta = b^2 - 4ac$

- **Trục đối xứng**: đường thẳng $x = -\\frac{b}{2a}$

- **Chiều mở**:
  - Nếu $a > 0$: parabol mở lên (có điểm cực tiểu)
  - Nếu $a < 0$: parabol mở xuống (có điểm cực đại)

## 1.3 Phương Trình Bậc Hai

**Phương trình bậc hai** là phương trình dạng:

$$ax^2 + bx + c = 0 \\quad (a \\neq 0)$$

### Cách Giải

**Bước 1**: Tính discriminant (biệt thức)

$$\\Delta = b^2 - 4ac$$

**Bước 2**: Phân tích theo giá trị của $\\Delta$:

- Nếu $\\Delta > 0$: phương trình có hai nghiệm phân biệt

  $$x_1 = \\frac{-b + \\sqrt{\\Delta}}{2a}, \\quad x_2 = \\frac{-b - \\sqrt{\\Delta}}{2a}$$

- Nếu $\\Delta = 0$: phương trình có nghiệm kép

  $$x = -\\frac{b}{2a}$$

- Nếu $\\Delta < 0$: phương trình vô nghiệm

### Ví Dụ 1

Giải phương trình: $x^2 - 5x + 6 = 0$

**Giải**:

- Xác định hệ số: $a = 1$, $b = -5$, $c = 6$
- Tính $\\Delta = (-5)^2 - 4(1)(6) = 25 - 24 = 1 > 0$
- Phương trình có hai nghiệm phân biệt:

  $$x_1 = \\frac{5 + 1}{2} = 3, \\quad x_2 = \\frac{5 - 1}{2} = 2$$

- **Kết luận**: $x = 2$ hoặc $x = 3$

### Ví Dụ 2

Giải phương trình: $2x^2 - 8x + 8 = 0$

**Giải**:

- Xác định hệ số: $a = 2$, $b = -8$, $c = 8$
- Tính $\\Delta = (-8)^2 - 4(2)(8) = 64 - 64 = 0$
- Phương trình có nghiệm kép:

  $$x = -\\frac{-8}{2(2)} = \\frac{8}{4} = 2$$

- **Kết luận**: $x = 2$ (nghiệm kép)

## 1.4 Định Lý Vieta

Nếu phương trình $ax^2 + bx + c = 0$ có hai nghiệm $x_1$ và $x_2$, thì:

$$x_1 + x_2 = -\\frac{b}{a}$$

$$x_1 \\cdot x_2 = \\frac{c}{a}$$

**Ứng dụng**: Lập phương trình bậc hai khi biết tổng và tích hai nghiệm.

**Ví dụ**: Lập phương trình bậc hai có hai nghiệm là $2$ và $3$.

$$x_1 + x_2 = 2 + 3 = 5, \\quad x_1 \\cdot x_2 = 2 \\cdot 3 = 6$$

Phương trình: $x^2 - 5x + 6 = 0$
""",

        "sample_linear_functions.mmd": """# Chương 2: Hàm Số Bậc Nhất

## 2.1 Định Nghĩa

**Hàm số bậc nhất** là hàm số được xác định bởi công thức:

$$y = ax + b$$

trong đó $a$ và $b$ là các hệ số thực và $a \\neq 0$.

## 2.2 Tính Chất

- **Tập xác định**: $\\mathbb{R}$ (toàn bộ tập số thực)
- **Chiều biến thiên**:
  - Nếu $a > 0$: hàm số đồng biến trên $\\mathbb{R}$
  - Nếu $a < 0$: hàm số nghịch biến trên $\\mathbb{R}$

## 2.3 Đồ Thị

Đồ thị của hàm số bậc nhất là một **đường thẳng** không song song và không trùng với các trục toạ độ.

- **Hệ số góc**: $a$ (độ dốc của đường thẳng)
- **Tung độ gốc**: $b$ (điểm cắt với trục $Oy$ là $(0, b)$)
- **Hoành độ gốc**: $-\\frac{b}{a}$ (điểm cắt với trục $Ox$ là $(-\\frac{b}{a}, 0)$)

## 2.4 Phương Trình Bậc Nhất

**Phương trình bậc nhất một ẩn** dạng:

$$ax + b = 0 \\quad (a \\neq 0)$$

**Nghiệm duy nhất**:

$$x = -\\frac{b}{a}$$

### Ví Dụ

Giải phương trình: $3x - 6 = 0$

**Giải**:

$$3x = 6$$
$$x = 2$$

**Kết luận**: $x = 2$
""",
    }
    
    os.makedirs("ocr_input", exist_ok=True)
    
    for filename, content in docs.items():
        filepath = Path("ocr_input") / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✓ Created: {filepath}")
    
    print(f"\n✅ Created {len(docs)} sample math documents in ocr_input/")

if __name__ == "__main__":
    create_sample_math_docs()
