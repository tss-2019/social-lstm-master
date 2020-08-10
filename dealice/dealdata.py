# demo = '1,3,d4,6574896765,5dsfjidshnjf'
# tmplist = filter(str.isdigit, demo)
# newlist = list(tmplist)
# print(newlist)
# print(tmplist)

f = open('../data/train/bingchang/demo.txt', 'r')
for line in f.readlines():
    temp = filter(str.isdigit, line)
    newlist = list(temp)
    print(newlist)


