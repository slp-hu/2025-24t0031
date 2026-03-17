import os
import tarfile

def extract_all_tar_gz(root_dir):
    """
    扫描 root_dir 下的所有 .tar.gz 文件，并全部解压到同一目录中。
    """
    for fname in sorted(os.listdir(root_dir)):
        if fname.endswith('.tar.gz'):
            tar_path = os.path.join(root_dir, fname)
            print(f"👉 正在解压: {tar_path}")
            try:
                with tarfile.open(tar_path, 'r:gz') as tar:
                    tar.extractall(path=root_dir)
                print(f"✅ 解压完成: {tar_path}")
            except Exception as e:
                print(f"❌ 解压失败 ({tar_path}): {e}")

if __name__ == "__main__":
    # 请根据实际情况调整以下路径
    melon_root = r"E:\melon\melon"
    extract_all_tar_gz(melon_root)
