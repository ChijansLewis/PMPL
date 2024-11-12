import re

def find_inner_quotes_in_text_field(file_paths):
    error_lines = {}

    # 正则表达式匹配 'text' 字段中的内容
    text_field_pattern = re.compile(r"'text':\s*'(.*?)'")

    for file_path in file_paths:
        file_errors = []
        with open(file_path, 'r', encoding='utf-8') as file:
            for line_number, line in enumerate(file, start=1):
                # 查找 'text' 字段的内容
                match = text_field_pattern.search(line)
                if match:
                    text_content = match.group(1)  # 获取 'text' 字段内容
                    # 检查内容是否包含双引号
                    if '"' in text_content:
                        file_errors.append(line_number)  # 记录含有内层双引号的行号
        error_lines[file_path] = file_errors
    
    return error_lines

# 指定文件路径
file_paths = ['dev.txt', 'test.txt', 'train.txt']
error_lines = find_inner_quotes_in_text_field(file_paths)

# 打印结果
for file, lines in error_lines.items():
    if lines:
        print(f"{file} 文件中存在内层双引号的行: {lines}")
    else:
        print(f"{file} 文件中没有内层双引号。")