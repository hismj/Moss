import os


def rename_images(folder_path):
    # 获取文件夹中的所有文件
    files = os.listdir(folder_path)
    # 过滤出图像文件（根据需要可以添加更多格式）
    img_files = [f for f in files if f.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

    # 按名称排序以确保顺序
    img_files.sort()

    # 遍历图像文件并重命名
    for index, file_name in enumerate(img_files):
        new_name = f'pred_{index + 1}{os.path.splitext(file_name)[1]}'  # 保留原始扩展名
        os.rename(os.path.join(folder_path, file_name), os.path.join(folder_path, new_name))

    print('rename done !!!!')

# 使用示例
folder_path = 'img_2_pred'  # 替换为图像文件夹的实际路径
rename_images(folder_path)
