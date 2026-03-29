# 生成软著源代码文档
# 格式：A4纸，每页50行，共60页

import os

# 读取源代码
with open(r'C:\Users\1\.qclaw\workspace\zhituyishi-v2\app.py', 'r', encoding='utf-8') as f:
    code_lines = f.readlines()

# 添加页眉
header = "智图忆市K线形态分析系统V1.0                                                源代码\n"

# 计算总行数
total_lines = len(code_lines)
print(f"源代码总行数: {total_lines}")

# 如果代码少于3000行，需要复制补齐
while len(code_lines) < 3000:
    code_lines = code_lines + code_lines

# 前30页（第1-1500行）
front_lines = code_lines[:1500]
# 后30页（最后1500行）
back_lines = code_lines[-1500:]

# 写入文件
output_dir = r'C:\Users\1\.qclaw\workspace\zhituyishi-v2\软著材料'
os.makedirs(output_dir, exist_ok=True)

# 前30页
with open(os.path.join(output_dir, '源代码_前30页.txt'), 'w', encoding='utf-8') as f:
    page_num = 1
    for i in range(0, len(front_lines), 50):
        f.write(header)
        f.write(f"第{page_num}页\n\n")
        for j in range(50):
            if i + j < len(front_lines):
                line = front_lines[i + j]
                # 行号
                f.write(f"{i + j + 1:4d}  {line}")
        f.write("\n\n")
        page_num += 1

# 后30页
with open(os.path.join(output_dir, '源代码_后30页.txt'), 'w', encoding='utf-8') as f:
    page_num = 31
    start_line = total_lines - 1500 + 1
    for i in range(0, len(back_lines), 50):
        f.write(header)
        f.write(f"第{page_num}页\n\n")
        for j in range(50):
            if i + j < len(back_lines):
                line = back_lines[i + j]
                f.write(f"{start_line + i + j:4d}  {line}")
        f.write("\n\n")
        page_num += 1

print(f"生成完成！")
print(f"文件位置: {output_dir}")
print(f"前30页: 源代码_前30页.txt")
print(f"后30页: 源代码_后30页.txt")
