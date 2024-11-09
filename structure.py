import os

def get_folder_size(folder_path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # 检查文件是否存在以避免错误
            if os.path.exists(fp):
                total_size += os.path.getsize(fp)
    return total_size

def display_folder_sizes(base_folder):
    for item in os.listdir(base_folder):
        item_path = os.path.join(base_folder, item)
        if os.path.isdir(item_path):
            folder_size = get_folder_size(item_path)
            print(f"文件夹: {item} 大小: {folder_size / (1024 * 1024):.2f} MB")  # 输出大小单位为 MB

# 使用示例
base_folder = '/home/STU/ljq/Projects/PMPL/'  # 替换为你想要查看的目录路径
display_folder_sizes(base_folder)
