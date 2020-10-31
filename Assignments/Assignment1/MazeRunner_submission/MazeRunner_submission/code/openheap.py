# Yan (bill) Gu
# yg369
# 189001028
#
# Heap for open set


class OpenHeap:
    def __init__(self):
        # initial as the start node, f = 0, coordinate is [0, 0]
        # self.heapList = [(0, [0, 0])]
        self.heapList = [0]
        self.currentSize = 0

    def build_heap(self, open_list):
        self.currentSize = len(open_list)
        self.heapList = open_list[:]
        # use merge sort to order the list
        self.heapList = self.merge_sort(self.heapList)

    def merge_sort(self, heap_list):
        if len(heap_list) <= 1:
            # When D&C to 1 element, just return it
            return heap_list
        mid = len(heap_list) // 2
        left = heap_list[:mid]
        right = heap_list[mid:]
        left = self.merge_sort(left)
        right = self.merge_sort(right)
        # conquer sub-problem recursively
        return self.merge(left, right)
        # return the answer of sub-problem

    @staticmethod
    def merge(left, right):
        result = []
        while left and right:
            if left[0] <= right[0]:
                result.append(left.pop(0))
            else:
                result.append(right.pop(0))
        if left:
            result += left
        if right:
            result += right
        print(result)
        return result

    def remove(self):
        del_item = self.heapList[0]
        del self.heapList[0]
        self.currentSize = self.currentSize - 1
        return del_item

    def insert(self, node):
        self.heapList.append(node)
        self.currentSize = self.currentSize + 1
        index_n = self.currentSize
        # insert order
        for i in range(2, index_n):
            if node <= self.heapList[index_n - i]:
                temp = self.heapList[index_n - i]
                self.heapList[index_n - i] = node
                self.heapList[index_n - i + 1] = temp

    def contains(self, coordinate):
        for i in range(0, self.currentSize):
            if self.heapList[i][1] == coordinate:
                return True
        return False