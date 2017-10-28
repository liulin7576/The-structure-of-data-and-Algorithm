import time
#顺序查找
def sequential_search(a_list,item):
    for i in range(len(a_list)):
        if item==a_list[i]:
            return True
    return False

test_list=[1,2,32,8,17,18,41,14,0]
start=time.time()
print(sequential_search(test_list,8))
print(sequential_search(test_list,16))
end=time.time()
print(end-start)

#二分查找
def binary_search(a_list,item):
    lower,upper=0,len(a_list)-1
    middle=(lower+upper)//2
    
    while lower<=upper:
        if middle<item:
            lower=middle+1
        elif middle>item:
            upper=middle
        else:
            return True
        middle=(lower+upper)//2 
    return False

start=time.time()
print(binary_search(test_list,8))
print(binary_search(test_list,16))
end=time.time()
print(end-start)


#hash Table（简单的取余数Python实现）
class HashTable:
    def __init__(self):
        self.size=11
        self.slots=[None]*self.size
        
    def put_data_in_slot(self,key,slot):#数据存放到slot里
        if self.slots[slot] == None:
            self.slots[slot] = key
            return True
        else:
            return False
    def rehash(self,old_hash):
        return (old_hash + 1) % self.size
        
    def put(self,key):#是否存入成功，没有则查找下一个
        slot = key % self.size
        result = self.put_data_in_slot(key,slot)
        while not result:
            slot = self.rehash(slot)
            result = self.put_data_in_slot(key,slot)

    def get(self,key):
        start_slot = key % self.size
        position = start_slot
        while self.slots[position] != None:
            if self.slots[position] == key:
                return True
            else:
                
                position = self.rehash(position)
                if position == start:
                    return False
        return False
             
hashtable=HashTable()
for i in test_list:
    hashtable.put(i)
start=time.time()
print(hashtable.get(8))
print(hashtable.get(16))
end=time.time()
print(end-start)























