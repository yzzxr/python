import random
from typing import List
from typing import Dict
from typing import Optional
from dataStructure import ListNode
import math

class Codec:
    def __init__(self) -> None:
        self.id = 0
        self.database: Dict[str, str] = {}

    def encode(self, longUrl: str) -> str:
        """Encodes a URL to a shortened URL."""
        self.id += 1
        self.database[self.id] = longUrl
        return "http://tinyurl.com/" + str(self.id)

    def decode(self, shortUrl: str) -> str:
        """Decodes a shortened URL to its original URL."""
        i = shortUrl.rfind("/")
        id_ = int(shortUrl[i + 1 :])
        return self.database[id_]


class Solution:
    def giveGem(self, gem: List[int], operations: List[List[int]]) -> int:
        for item in operations:
            gem[item[1]] += gem[item[0]] // 2
            gem[item[0]] -= gem[item[0]] // 2

        return max(gem) - min(gem)

    def queensAttacktheKing(
        self, queens: List[List[int]], king: List[int]
    ) -> List[List[int]]:
        result: List[List[int]] = []

        for direction in [
            [-1, -1],
            [-1, 0],
            [-1, 1],
            [0, -1],
            [0, 1],
            [1, -1],
            [1, 0],
            [1, 1],
        ]:
            temp: List[int] = list(king)  # 创建一个新列表必须使用list()函数
            while temp[0] in range(0, 8) and temp[1] in range(0, 8):
                temp[0] += direction[0]
                temp[1] += direction[1]
                if temp in queens:
                    result.append(temp)
                    break

        return result

    def rob(self, nums: List[int]) -> int:
        """打家劫舍，动态规划"""
        size: int = len(nums)

        if size == 0:
            return 0

        if size == 1:
            return nums[0]

        first, second = nums[0], max(nums[0], nums[1])
        for i in range(2, size):
            first, second = second, max(first + nums[i], second)

        return second

    # 双指针法
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        n = len(nums)  # 一定要一开始就把数组长度写出来***************至关重要
        nums.sort()
        ans: List[List[int]] = list()

        # 枚举 a
        for first in range(n - 2):
            # 需要和上一次枚举的数不相同
            if first > 0 and nums[first] == nums[first - 1]:
                continue
            # c 对应的指针初始指向数组的最右端
            third = n - 1
            target = -nums[first]
            # 枚举 b
            for second in range(first + 1, n - 1):
                # 需要和上一次枚举的数不相同
                if second > first + 1 and nums[second] == nums[second - 1]:
                    continue
                # 需要保证 b 的指针在 c 的指针的左侧
                while second < third and nums[second] + nums[third] > target:
                    third -= 1
                # 如果指针重合，随着 b 后续的增加
                # 就不会有满足 a+b+c=0 并且 b<c 的 c 了，可以退出循环
                if second == third:
                    break
                if nums[second] + nums[third] == target:
                    ans.append([nums[first], nums[second], nums[third]])

        return ans

    def threeSum2(self, nums: List[int]) -> List[List[int]]:
        """优化后的三数之和"""
        nums.sort()
        ans = []
        n = len(nums)
        for i in range(n - 2):
            x = nums[i]
            if i > 0 and x == nums[i - 1]:  # 跳过重复数字
                continue
            if x + nums[i + 1] + nums[i + 2] > 0:  # 优化一
                break
            if x + nums[-2] + nums[-1] < 0:  # 优化二
                continue
            j = i + 1
            k = n - 1
            while j < k:
                s = x + nums[j] + nums[k]
                if s > 0:
                    k -= 1
                elif s < 0:
                    j += 1
                else:  # sum == 0
                    ans.append([x, nums[j], nums[k]])

                    j += 1
                    while j < k and nums[j] == nums[j - 1]:  # 跳过重复数字
                        j += 1
                    k -= 1
                    while k > j and nums[k] == nums[k + 1]:  # 跳过重复数字
                        k -= 1
        return ans

    def threeSumSelf(self, nums: List[int]) -> List[List[int]]:
        """自实现的三数之和"""
        n = len(nums)
        nums.sort()
        ans = []
        for a in range(n - 2):
            if a > 0 and nums[a - 1] == nums[a]:
                continue
            if nums[a] + nums[a + 1] + nums[a + 2] > 0:
                break
            if nums[a] + nums[-2] + nums[-1] < 0:
                continue

            b = a + 1
            c = n - 1

            while b < c:
                sum = nums[a] + nums[b] + nums[c]

                if sum > 0:
                    c -= 1
                elif sum < 0:
                    b += 1
                else:
                    ans.append([nums[a], nums[b], nums[c]])
                    b += 1
                    while b < c and nums[b] == nums[b - 1]:
                        b += 1
                    c -= 1
                    while c > b and nums[c] == nums[c + 1]:
                        c -= 1

        return ans

    def merge_two_array(self, nums1: List[int], nums2: List[int]) -> List[int]:
        """合并两个有序数组"""
        n1 = len(nums1)
        n2 = len(nums2)
        a, b = 0, 0

        result: List[int] = []

        while a < n1 and b < n2:
            if nums1[a] > nums2[b]:
                result.append(nums2[b])
                b += 1
            elif nums1[a] < nums2[b]:
                result.append(nums1[a])
                a += 1
            else:
                result.append(nums1[a])
                result.append(nums2[b])
                a += 1
                b += 1

        if a == n1:
            result += nums2[b:]
        if b == n2:
            result += nums1[a:]

        return result

    def fourSum(self, nums: List[int], target: int) -> List[List[int]]:
        """四数之和"""
        n = len(nums)
        nums.sort()
        result: List[List[int]] = []

        if n < 4:
            return result

        for a in range(n - 3):
            # 同一重循环中，如果当前元素与上一个元素相同，则跳过当前元素。
            if a > 0 and nums[a] == nums[a - 1]:
                continue
            if nums[a] + nums[a + 1] + nums[a + 2] + nums[a + 3] > target:
                break
            if nums[a] + nums[n - 3] + nums[n - 2] + nums[n - 1] < target:
                continue

            for b in range(a + 1, n - 2):
                # 同一重循环中，如果当前元素与上一个元素相同，则跳过当前元素。
                if b > a + 1 and nums[b] == nums[b - 1]:
                    continue
                if nums[a] + nums[b] + nums[b + 1] + nums[b + 2] > target:
                    break
                if nums[a] + nums[b] + nums[n - 2] + nums[n - 1] < target:
                    continue

                c = b + 1
                d = n - 1
                while c < d:
                    sum = nums[a] + nums[b] + nums[c] + nums[d]
                    if sum > target:
                        d -= 1
                    elif sum < target:
                        c += 1
                    else:
                        result.append([nums[a], nums[b], nums[c], nums[d]])
                        c += 1
                        while c < d and nums[c - 1] == nums[c]:
                            c += 1
                        d -= 1
                        while d > c and nums[d + 1] == nums[d]:
                            d -= 1

        return result

    def numDistinct(self, s: str, t: str) -> int:
        """动态规划 子字符串的个数"""
        m = len(s)
        n = len(t)

        if m < n:
            return 0

        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            dp[i][n] = 1

        # 反向遍历
        for i in range(m - 1, -1, -1):
            for j in range(n - 1, -1, -1):
                if s[i] == t[j]:
                    dp[i][j] = dp[i + 1][j + 1] + dp[i + 1][j]
                else:
                    dp[i][j] = dp[i + 1][j]

        return dp[0][0]

    def twoSum(self, nums: List[int], target: int) -> List[int]:
        m: Dict[int, int] = {}

        for i, num in enumerate(nums):
            if target - num in m:
                return [i, m[target - num]]
            else:
                m[target - num] = i

        return []

    def addTwoNumbers(
        self, l1: Optional[ListNode], l2: Optional[ListNode]
    ) -> Optional[ListNode]:
        head: ListNode = None
        tail: ListNode = None

        carry = 0  # 进位
        while l1 is not None or l2 is not None:
            n1 = l1.val if l1 is not None else 0
            n2 = l2.val if l2 is not None else 0
            sum = n1 + n2 + carry

            if head is None:
                head = ListNode(sum % 10)
                tail = head
            else:
                tail.next = ListNode(sum % 10)
                tail = tail.next

            carry = sum // 10  # < 1

            if l1 is not None:
                l1 = l1.next
            if l2 is not None:
                l2 = l2.next

        if carry > 0:
            tail.next = ListNode(carry)

        return head

    def threeSumClosest(self, nums: List[int], target: int) -> int:
        nums.sort()
        n = len(nums)
        result: int = 10**7  # 初始值必须足够大

        for a in range(n - 2):
            if a > 0 and nums[a] == nums[a - 1]:
                continue
            # if nums[a] + nums[a + 1] + nums[a + 2] > target:
            #     return nums[a] + nums[a + 1] + nums[a + 2]
            # if nums[a] + nums[n - 2] + nums[n - 1] < target:
            #     return nums[a] + nums[n - 2] + nums[n - 1]

            b = a + 1
            c = n - 1
            while b < c:
                sum = nums[a] + nums[b] + nums[c]
                if sum == target:
                    return target
                elif abs(sum - target) < abs(result - target):
                    result = sum  # 只起到更新的作用

                if sum > target:
                    c -= 1
                    while b < c and nums[c] == nums[c + 1]:
                        c -= 1
                else:
                    b += 1
                    while b < c and nums[b] == nums[b - 1]:
                        b += 1

        return result

    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        if l1 is None:
            return l2
        elif l2 is None:
            return l1

        elif l1.val < l2.val:
            l1.next = self.mergeTwoLists(l1.next, l2)
            return l1
        else:
            l2.next = self.mergeTwoLists(l1, l2.next)
            return l2

        # 非递归解法
        # head: ListNode = None
        # tail: ListNode = None

        # if list1 is None:
        #     return list2
        # if list2 is None:
        #     return list1

        # while list1 is not None and list2 is not None:

        #     if head is None:
        #         if list1.val < list2.val:
        #             head = ListNode(list1.val)
        #             tail = head
        #             list1 = list1.next
        #         elif list1.val > list2.val:
        #             head = ListNode(list2.val)
        #             tail = head
        #             list2 = list2.next
        #         else:
        #             head = ListNode(list1.val)
        #             head.next = ListNode(list2.val)
        #             tail = head
        #             tail = tail.next
        #             list1 = list1.next
        #             list2 = list2.next
        #     else:
        #         if list1.val < list2.val:
        #             tail.next = ListNode(list1.val)
        #             tail = tail.next
        #             list1 = list1.next
        #         elif list1.val > list2.val:
        #             tail.next = ListNode(list2.val)
        #             tail = tail.next
        #             list2 = list2.next
        #         else:
        #             tail.next = ListNode(list1.val)
        #             tail.next.next = ListNode(list2.val)
        #             tail = tail.next.next
        #             list1 = list1.next
        #             list2 = list2.next

        # if list1 is None:
        #     tail.next = list2
        # if list2 is None:
        #     tail.next = list1

        # return head

    def splitNum(self, num: int) -> int:
        s = str(num)
        l = list(s)
        l.sort()
        n = len(l)

        a = ""
        b = ""

        for i in range(n):
            if i % 2 == 0:
                a += l[i]
            else:
                b += l[i]

        a = int(a)
        b = int(b)

        return a + b

    def reverse_list(self, head: ListNode) -> ListNode:
        """逆序单链表，递归实现"""
        if head is None or head.next is None:
            return head

        new_head = self.reverse_list(head.next)
        head.next.next = head
        head.next = None

        return new_head

    def reverse_list1(self, head: ListNode) -> ListNode:
        """非递归实现逆序链表"""
        if head is None or head.next is None:
            return

        pre: ListNode = head
        current: ListNode = head.next
        next: ListNode = None

        while current is not None:
            next = current.next
            current.next = pre
            pre = current
            current = next

        head.next = None
        head = pre

        return head

    @staticmethod
    def reverse_list2(head: ListNode) -> ListNode:
        """头插法"""
        new_head = ListNode(head.val)
        head = head.next

        while head is not None:
            temp = ListNode(head.val, new_head)
            new_head = temp
            head = head.next

        return new_head

    @staticmethod
    def maxArea(
        h: int, w: int, horizontalCuts: List[int], verticalCuts: List[int]
    ) -> int:
        """最大蛋糕"""
        horizontalCuts.sort()
        horizontalCuts.append(h)
        horizontalCuts.insert(0, 0)

        verticalCuts.sort()
        verticalCuts.append(w)
        verticalCuts.insert(0, 0)

        l_h = len(horizontalCuts)
        l_v = len(verticalCuts)

        h_gaps: List[int] = []
        for i in range(l_h - 1):
            gap = horizontalCuts[i + 1] - horizontalCuts[i]
            h_gaps.append(gap)

        v_gaps: List[int] = []
        for i in range(l_v - 1):
            gap = verticalCuts[i + 1] - verticalCuts[i]
            v_gaps.append(gap)

        result = (max(h_gaps) * max(v_gaps)) % (10**9 + 7)

        return result

    @staticmethod
    def countPoints(points: List[List[int]], queries: List[List[int]]) -> List[int]:
        def point_is_in_circle(circle: List[int], point: List[int]) -> bool:
            if (point[0] - circle[0]) ** 2 + (point[1] - circle[1]) ** 2 <= circle[
                2
            ] ** 2:
                return True
            else:
                return False

        query_count = len(queries)
        answer: List[int] = []

        for i in range(query_count):
            count: int = 0
            for point in points:
                if point_is_in_circle(queries[i], point):
                    count += 1
            answer.append(count)

        return answer

    def differenceOfSums(self, n: int, m: int) -> int:
        can = []
        cant = []

        for x in range(1, n + 1):
            if x % m == 0:
                can.append(x)
            else:
                cant.append(x)

        return sum(cant) - sum(can)

    def maxIncreaseKeepingSkyline(self, grid: List[List[int]]) -> int:
        n = len(grid)
        row_maxs = []
        col_maxs = []

        for i in range(n):
            row_maxs.append(max(grid[i]))

        for j in range(n):
            temp = grid[0][j]
            for i in range(1, n):
                if grid[i][j] > temp:
                    temp = grid[i][j]
            col_maxs.append(temp)

        res = 0
        for i in range(n):
            for j in range(n):
                res += min(row_maxs[i], col_maxs[j]) - grid[i][j]

        return res

    def isStrictlyPalindromic(self, n: int) -> bool:
        """是否严格回文"""

        def is_huiwen(s: str) -> bool:
            n = len(s)

            for i in range(n // 2):
                if s[i] == s[n - 1 - i]:
                    continue
                else:
                    return False

            return True

        for i in range(2, n - 1):
            res = ""
            quotient, remainder = divmod(n, i)
            res += str(remainder)
            while quotient >= i:
                quotient, remainder = divmod(quotient, i)
                res += str(remainder)
            res += str(remainder)
            res = res[::-1]

            if not is_huiwen(res):
                return False

        return True

    def insertGreatestCommonDivisors(
        self, head: Optional[ListNode]
    ) -> Optional[ListNode]:
        p = head
        q = p.next
        while p is not None and q is not None:
            temp = math.gcd(p.val, q.val)
            temp_node = ListNode(temp, q)
            p.next = temp_node

            p = q
            q = q.next

        return head

    def minOperations(self, boxes: str) -> List[int]:
        """最小操作数"""
        # n = len(boxes)

        # answer: List[int] = []

        # for i in range(n):
        #     count = 0
        #     for j in range(0, i):
        #         if boxes[j] == '0':
        #             continue
        #         else:
        #             count += i - j
        #     for k in range(i + 1, n):
        #         if boxes[k] == '0':
        #             continue
        #         else:
        #             count += k - i

        #     answer.append(count)

        # return answer

        """第二种解法,简便"""
        # res = []
        # for i in range(len(boxes)):
        #     s = sum(abs(j - i) for j, c in enumerate(boxes) if c == '1')
        #     res.append(s)
        # return res

        left, right, operations = int(boxes[0]), 0, 0
        for i in range(1, len(boxes)):  # 第一重循环确定初始left, right和 operation
            if boxes[i] == "1":
                right += 1
                operations += i - 0

        res = [operations]

        for i in range(1, len(boxes)):
            operations += left - right  # 关键
            if boxes[i] == "1":
                left += 1
                right -= 1
            res.append(operations)

        return res

    def singleNumber(self, nums: List[int]) -> int:
        """只出现一次的数, 异或直接秒杀"""
        res = 0
        for item in nums:
            res ^= item
        return res

    def areTwoLinkListIntersect(self, l1: ListNode, l2: ListNode) -> bool:
        """判断两链表是否相交,判断最后一个节点是否相同即可"""
        while l1.next is not None:
            l1 = l1.next

        while l2.next is not None:
            l2 = l2.next

        return l1 is l2

    def getIntersectionNode(
        self, headA: ListNode, headB: ListNode
    ) -> Optional[ListNode]:
        """获取两链表第一个相交节点"""

        ids1 = set()
        while headA is not None:
            ids1.add(id(headA))
            headA = headA.next

        while headB is not None:
            if id(headB) in ids1:
                return headB
            headB = headB.next

        return None

    def getIntersectionNode1(
        self, headA: ListNode, headB: ListNode
    ) -> Optional[ListNode]:
        """获取两链表第一个相交节点,方法二, 双指针法 O(m+n) O(1)"""
        pa = headA
        pb = headB

        while pa is not pb:
            if pa is None:
                pa = headB
            else:
                pa = pa.next

            if pb is None:
                pb = headA
            else:
                pb = pb.next

        return pa

    def deleteNode(self, head: ListNode, val: int) -> ListNode:
        """删除链表节点"""
        if head is None:
            return head

        if head.val == val:
            return head.next

        pre = head
        cur = head.next

        while cur is not None:
            if cur.val == val:
                pre.next = cur.next
                break
            pre = pre.next
            cur = cur.next

        return head

    def reverse_list3(self, head: ListNode):
        """原地逆序单链表, 类似于放书"""
        if head is None:
            return head

        q = None
        while head is not None:
            p = head
            head = p.next
            p.next = q
            q = p
        head = p

        return head

    def insertion_sort(self, l: List[int]) -> None:
        """
        插入排序, 维护一段已经排好序的片段，新元素不断与这个里面的元素比较
        新元素大则不变，新元素小则将该位置元素向后移动一位, 稳定的排序
        """
        n = len(l)

        if n == 0 or n == 1:
            return

        for i in range(1, n):
            temp = l[i]  # 腾出空位给前面的元素
            j = i  # 新变量j 向前迭代
            while j > 0 and l[j - 1] > temp:
                l[j] = l[j - 1]
                j -= 1
            l[j] = temp

    def sort_link_list(self, head: ListNode) -> None:
        """排序单链表, 插入排序思想， 交换链表元素值"""
        if head is None:
            return

        cur = head.next
        while cur is not None:
            p = head
            x = cur.val

            while p is not None and p.val <= x:
                p = p.next

            while p is not None and p is not cur:
                y = p.val
                p.val = x
                x = y
                p = p.next

            cur.val = x
            cur = cur.next

    def bubble_sort(self, l: list[int]):
        """改进的起泡排序算法"""
        n = len(l)
        for _ in range(n):
            found = False   # 没发现顺序颠倒，标记是否已经是有序的了，省去不必要的比较和交换操作
            for j in range(0, n - 1):
                if l[j + 1] < l[j]:
                    l[j], l[j + 1] = l[j + 1], l[j]
                    found = True
            if not found:
                break
            
    def quick_sort(self, ints: list[int], left: int, right: int) -> None:
        if right <= left:
            return  # 递归出口
        
        i = left
        j = right
        
        pivot = ints[i]
        while i < j:
            while i < j and ints[j] >= pivot:
                j -= 1
            if i < j:
                ints[i] = ints[j]
                i += 1 
            
            while i < j and ints[i] <= pivot:
                i += 1
            if i < j:
                ints[j] = ints[i]
                j -= 1
        assert i == j, "Error!"
        ints[i] = pivot

        self.quick_sort(ints, left, i - 1)
        self.quick_sort(ints, i + 1, right)

    def canPlantFlowers(self, f: list[int], n: int) -> bool:
        
        l = len(f)
        count = 0
        for i in range(l):
            if i == 0 or f[i - 1] == 0 and f[i] == 0 and i == l - 1 or f[i + 1] == 0:
                count += 1
                f[i] = 1
        
        return count >= n
        

    def findPoisonedDuration(self, timeSeries: List[int], duration: int) -> int:
        '''提莫打艾希'''
        n = len(timeSeries)
        
        result = 0
        
        for i in range(n - 1):
            if timeSeries[i] <= timeSeries[i + 1] <= timeSeries[i] + duration:
                result += timeSeries[i + 1] - timeSeries[i]
            else:
                result += duration
        result += duration
        
        return result
        







