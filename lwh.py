import os
#
# # 读取20个人的轨迹来测试一下
# f1 = open('./data/train/biwi/biwi_hotel.txt', 'r')
# f2 = open('./data/train/biwi/biwi_hotel_demo.txt', 'w')
# for i in range(400):
#     line = f1.readline()
#     # print(line)
#     f2.write(line)
# f2.close()
# f1.close()

# 测试创建文件代码的问题
def create_directories(base_folder_path, folder_list):
    # create folders using a folder list and path
    for folder_name in folder_list:
        directory = os.path.join(base_folder_path, folder_name)
        if not os.path.exists(directory):
            os.makedirs(directory)

