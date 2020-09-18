# import numpy as np
# import matplotlib.pyplot as plt
#
# np.set_printoptions(suppress=True)
# r = 1.0
# a, b = (0., 0.)
# theta = np.arange(0, 2 * np.pi, 0.001)
# x = np.array(a + r * np.cos(theta))
# y = np.array(b + r * np.sin(theta))
# fig = plt.figure()
# axes = fig.add_subplot(111)
# axes.plot(x, y)
# axes.axis('equal')
# plt.show()
# z = np.array([y, x])
#
# np.savetxt('./yuan2.txt', x, fmt='%f', delimiter='\t')
# import numpy as np
#
# a = np.ones(6285)
# np.savetxt('./xuhao.txt', a, fmt='%d', delimiter='\t')

# 有噪声
# import random
# mu = 0
# sigma =0.001
# for i in range(theta.size):
#     x[i] += random.gauss(mu,sigma)
#     y[i] += random.gauss(mu,sigma)
#
# # 画出
# plt.plot(x,y,linestyle='',marker='.')
# plt.show()
