n = int(input())
map_list = [[0 for _ in range(n)] for _ in range(n)]
k = int(input())
for _ in range(k) :
    x, y = map(int, input().split())
    map_list[x-1][y-1] = 2
l = int(input())
from collections import deque
directions = deque([])
for _ in range(l) :
    directions.append(list(map(str, input().split())))

map_list[0][0] = 1
snake_head = [0,0,2]
snake_body = deque([[0,0]])

dy = [-1, 0, 1, 0]
dx = [0, -1, 0, +1]

time = 0

def change_direction(direction) :
    if direction == 'L' :
        snake_head[2] -= 1 if snake_head[2] > 0 else -3
    elif direction == 'D' :
        snake_head[2] += 1 if snake_head[2] < 3 else -3

def move() :
    global time
    time += 1
    snake_head[0] += dx[snake_head[2]]
    snake_head[1] += dy[snake_head[2]]
    if snake_head[0] >= n or snake_head[0] < 0 or snake_head[1] >= n or snake_head[1] < 0 : return 0
    if map_list[snake_head[0]][snake_head[1]] == 1 :
        map_list[snake_head[0]][snake_head[1]] = 3
        return False
    elif map_list[snake_head[0]][snake_head[1]] == 0 :
        tail = snake_body.popleft()
        map_list[tail[0]][tail[1]] = 0
    snake_body.append([snake_head[0], snake_head[1]])
    map_list[snake_head[0]][snake_head[1]] = 1
    return True

while 0<=snake_head[0]<n and 0<=snake_head[1]<n :
    if map_list[snake_head[0]][snake_head[1]] == 3 : break
    if directions :
        change = directions.popleft()
        for i in range(int(change[0])-time) :
            if not move() : break
        change_direction(change[1])
    else :
        move()

print(time)