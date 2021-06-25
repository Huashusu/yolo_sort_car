import os
import shutil
from distutils.core import setup

from Cython.Build import cythonize

dirPath = "yolo.py"
filePath3 = "build/"
# 1、文件加密
setup(ext_modules=cythonize([dirPath]))
# 2、将加密的文件移至对应目录下
files_1 = os.listdir(filePath3)

for files_1_temp in files_1:
    if "lib" in files_1_temp:
        files_1 = files_1_temp
        print(files_1)

files_2 = os.listdir(filePath3 + files_1)[0]
so_file = filePath3 + files_1 + "/" + files_2
print(so_file)

# 文件移动或拷贝
shutil.copy(so_file, "./")

# 3、删除原文件和生成的附属文件夹
# files2 = os.listdir("./")
# for file in files2:
#     if file == dirPath or file.endswith(".c"):
#         # 判断文件是否存在
#         if (os.path.exists(file)):
#             os.remove(file)
#             print('移除后test 目录下有文件：%s' % file)
#         else:
#             print("要删除的文件不存在！")

# #删除附属文件夹
# try:
#     shutil.rmtree(filePath3)
# except Exception as ex:
#     print("错误信息："+str(ex))#提示：错误信息，目录不是空的

print("删除完成")
print("加密完成")
