
# 读取20个人的轨迹来测试一下
f1 = open('./data/train/biwi/biwi_hotel.txt', 'r')
f2 = open('./data/train/biwi/biwi_hotel_demo.txt', 'w')
for i in range(400):
    line = f1.readline()
    # print(line)
    f2.write(line)
f2.close()
f1.close()