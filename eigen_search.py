#-*- coding:utf-8 -*-
'给定一个元素是数字的二维矩阵和一个整数列表，找出所有同时在矩阵和数组中出现的整数。'
'整数必须按照顺序，通过相邻的单元格内的数字构成，方向不限，'
'同一个单元格内的数字不允许被重复使用。'


class Solution:
    def __init__(self):
        self.pos = list()  #
        self.n = self.m = 0  # 用来储存行和列的数目
        self.flag = []  # 用来标志最终结果
        self.l = 0  # 数字长度
        self.target = ""  # 目标要构建的数字
        self.direction = [[1, 0], [-1, 0], [0, 1], [0, -1]]  # 可以选的方向

    def search(self, Matrix, x, y, s):
        if s == self.target:
            self.flag = True
            return  # 如果构建中的字符串达到目标要构建的数字
        if len(s) >= self.l:
            return  # 如果构建的字符的长度大了，就返回，防止无限增加
        self.pos[x][y] = True  # 把当前位置标记为已搜索，防止再次搜索。
        for d in self.direction:  # 遍历所有方向，上下左右
            if 0 <= x + d[0] <= self.n - 1 and 0 <= y + d[1] <= self.m - 1 and not self.pos[x + d[0]][y + d[1]] and \
                    Matrix[x + d[0]][y + d[1]] == self.target[len(s)]:  # 只有当下个节点没有超过边界并且没有被访问过并且是下一个必须的组成数字的下一个字母时，我们才调用递归
                self.search(Matrix, x + d[0], y + d[1], s + Matrix[x + d[0]][y + d[1]])  # 构建中的数字需要加上下一个字母，同时更新位置
                if self.flag:
                    return  # 如果找到了该数字，直接返回
        self.pos[x][y] = False  # 把当前标为未访问

    def find_integers(self, Matrix, numbers):
        self.res = []
        numbers = [str(i) for i in numbers]
        Matrix = [[str(i) for i in j] for j in Matrix ]
        for num in numbers:
            num = str(num)
            self.n = len(Matrix)  # 储存行数
            if self.n == 0:
                return False
            self.m = len(Matrix[0])  # 储存列数
            if self.m == 0:
                return False
            self.l = len(num)  # 储存数字长度
            if self.l == 0 or self.l > self.n * self.m:
                return False  # 如果当前数字比所有字母连一起还长
            self.target = num  # 储存目标数字
            self.pos = [[False for col in range(self.m)] for row in range(self.n)]  # 构建一个检测有没有被访问的表
            for i in range(self.n):
                for j in range(self.m):
                    if Matrix[i][j] == num[0]:  # 如果当前字母等于第一个字母，开始递归
                        self.search(Matrix, i, j, Matrix[i][j])
            self.res.append(self.flag)
            self.flag = False
        output = []
        for i in range(len(self.res)):
            if self.res[i]:
                output.append(int(numbers[i]))
        print(output)

numbers =  [123, 895, 119, 1037]
Matrix =  [
  [1, 2, 3, 4],
  [3, 5, 9, 8],
  [8, 0, 3, 7],
  [6, 1, 9, 2]
]

Solution().find_integers(Matrix, numbers)
