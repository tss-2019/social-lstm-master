# demo = '1,3,d4,6574896765,5dsfjidshnjf'
# tmplist = filter(str.isdigit, demo)
# newlist = list(tmplist)
# print(newlist)
# print(tmplist)

# f = open('../data/train/bingchang/demo.txt', 'r')
# for line in f.readlines():
#     temp = filter(str.isdigit, line)
#     newlist = list(temp)
#     print(newlist)

# import os
# f = open('../data/train/bingchang/100row.txt')
# lines = f.readlines()
# f.close()
# newname = '../data/train/bingchang/lwh_1.txt'
# if os.path.exists(newname):
#     os.remove(newname)
#
# for line in lines:
#     newtxt = line.replace('' '')
# newfile = open('../data/train/bingchang/lwh_1.txt', 'a')
# newfile.write(newtxt)
# newfile.close()

# f = open('../data/train/bingchang/100.txt')
# print(f.readlines())

# 对数据进行归一化处理
# import numpy as np
# import pandas as pd
#
# # sep设置参数会读取为4列
# df = pd.read_table('../data/train/bingchang/100.txt', sep=' ', names = ['A','B', 'C', 'D'])
# # print(df)
# # print(df[['C']])
# #####归一化函数#####
# max_min_scaler = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))
#
#
# df[['C']] = df[['C']].apply(max_min_scaler)
# df[['D']] = df[['D']].apply(max_min_scaler)
#
# # print(df)
# # 安装官网文档的写法保留小数会失效
# # df.round(1)
# print(df)
#
# df.to_csv('../data/train/bingchang/100.csv', header=0, index=0)

# # 将csv转为txt
# import pandas as pd
#
# data = pd.read_csv('../data/train/bingchang/100.csv')
# with open('../data/train/bingchang/100_1.txt', 'a+') as f:
#     for line in data.values:
#         f.write((str(line[0]) + ' ' + str(line[1]) + ' ' + str(line[2]) + ' ' + str(line[3]) + '\n'))

# f = open('../data/train/ceshi/100_1.txt')
# print(f.readlines())

