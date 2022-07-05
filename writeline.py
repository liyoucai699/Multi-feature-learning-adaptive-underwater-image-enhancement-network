# import re
# reg=re.compile(r"Avg fI val SSIM: ")
# # match=reg.search("字符串")
# # print match.group(0)
#
# data = []
# for line in open("checkpoints/realword/log_file.txt", "r"): #设置文件对象并读取每一行文件
#     match = reg.search(line)
#     data.append(line)               #将每一行文件加入到list中
import re  # 导入模块
import numpy as np  # 导入numpy模块
import matplotlib.pyplot as plt  # 导入matplotlib模块


def get_score(fn, ks):
    filename = fn
    keystr = ks
    count = 0
    tmpy = 0
    y = []
    pos = []
    with open(filename, 'r') as f:  # 打开文件用open，类似C语言，得到一个文件指针f
        for lines in f.readlines():  # readlines是读取文件的所有行，用for语句就是循环读，lines就代表当前这一行的内容
            tmp = [str(i) for i in lines.split(',', -1)]  # lines.split即把lines的内容通过逗号分开，分别储存在tmp中
            for a in tmp:  # 然后再把tmp中的每一个字符串拿出来检查
                m = re.search(keystr, a)  # re.search()是关键字搜索功能（即我们上面的关键字），如果没搜索到就返回None
                if m != None:  # 所以如果m不等于None，就说明这个字符串有我需要信息
                    n = [str(i) for i in a.split(':', -1)]  # 然后通过冒号把这个字符串再拆解，冒号后面就是数字
                    nn = [str(i) for i in n[1].split(' ', -1)]  # #然后再把后面的字符串通过空格再拆解，得到两个数字，和一个换行符
                    # nn.remove('\n')  # 删掉换行符
                    count = count + 1  # 记录有效数据个数，然后下面通过int（）函数把它转为数字类型
                    # tmpy = (int(nn[1]))
                    # if len(nn) == 1:
                    #     tmpy = (int(nn[1], 16))
                    # else:
                    #     tmpy = (int(nn[0], 16)) + (int(nn[1], 16)) * 256
                    # y.append(tmpy)  # 保存到y数组中，append是添加一个元素的意思
                    y.append(float(nn[1]))  # 保存到y数组中，append是添加一个元素的意思
        print(y)  # 然后把所有y打印出来，就是上面效果展示的左边
        print(count)
        return y, count

filename_1 = 'checkpoints/realword-2/log_file.txt'  # txt文件名
filename_2 = 'checkpoints/realword-att/log_file.txt'  # txt文件名
filename_3 = 'checkpoints/realword-dul2/log_file.txt'  # txt文件名
filename_4 = 'checkpoints/realword-class/log_file.txt'  # txt文件名
keystr = r"Avg fI val SSIM: "  # 关键字是AttrData： ，因为目的是把它后面的数字提取出来

y_1, count1 = get_score(filename_1, keystr)
y_2, count2 = get_score(filename_2, keystr)
y_3, count3 = get_score(filename_3, keystr)
y_4, count4 = get_score(filename_4, keystr)

x_1 = np.linspace(0, count1 - 1, len(y_1))  # 然后给x赋值，从0到count，x每次递增1
x_2 = np.linspace(0, count2 - 1, len(y_2))  # 然后给x赋值，从0到count，x每次递增1
x_3 = np.linspace(0, count3 - 1, len(y_3))  # 然后给x赋值，从0到count，x每次递增1
x_4 = np.linspace(0, count4 - 1, len(y_4))  # 然后给x赋值，从0到count，x每次递增1
plt.plot(x_1, y_1)  # 把提取出来的作为y值，x轴递增1来展示y值的变化
plt.plot(x_2, y_2)  # 把提取出来的作为y值，x轴递增1来展示y值的变化
plt.plot(x_3, y_3)  # 把提取出来的作为y值，x轴递增1来展示y值的变化
plt.plot(x_4, y_4)  # 把提取出来的作为y值，x轴递增1来展示y值的变化
plt.show()  # 展示图像，右边所示

