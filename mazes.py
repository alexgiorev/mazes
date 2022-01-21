import itertools
import math
import networkx
import pdb

from PIL import Image, ImageDraw

class Rect:
    LEFT = (-1,0)
    RIGHT = (1,0)
    UP = (0,-1)
    DOWN = (0,1)
    
    def __init__(self, top_left,width,height):
        self.top_left = top_left
        self.width = width
        self.height = height

    def __eq__(self,other):
        return (type(other) is type(self) and
                (self.top_left,self.width,self.height) ==
                (other.top_left,other.width,other.height))
    
    def __hash__(self):
        return hash((self.top_left,self.width,self.height))

    def __repr__(self):
        return f"Rect({self.top_left},{self.width},{self.height})"

    def move(self, direction, amount=1):
        x,y = self.top_left
        x += direction[0]*amount
        y += direction[1]*amount
        return Rect((x,y),self.width,self.height)

    @property
    def top_right(self):
        x,y = self.top_left
        return (x+self.width-1, y)
    
    @property
    def bottom_left(self):
        x,y = self.top_left
        return (x, y+self.height-1)
    
    @property
    def bottom_right(self):
        x,y = self.top_left
        return (x+self.width-1, y+self.height-1)

    @property
    def midpoint(self):
        (tx,ty),(bx,by) = (self.top_left, self.bottom_right)
        return (tx+bx)//2, (ty+by)//2

class MazeImage:
    TUNNEL_COLOR = (255,255,255)
    WALL_COLOR = (136,170,136)
    CIRCLE_BORDER_COLOR = (0,0,255)
    CIRCLE_INTERNALS_COLOR = (255,215,0)
    COLORS = (TUNNEL_COLOR,
              WALL_COLOR,
              CIRCLE_BORDER_COLOR,
              CIRCLE_INTERNALS_COLOR)

    def __init__(self,img):
        self.img = img
        
    def in_tunnel(self,xy):
        color = self.img.getpixel(xy)
        # return math.dist(color, self.TUNNEL_COLOR) <= 5
        return color == (255,255,255)

    def junct_neighbor(self, junct, direction):
        cursor = self._get_cursor(junct,direction)
        if not self.in_tunnel(cursor.midpoint):
            return None
        while True:
            if not self.in_tunnel(cursor.midpoint):
                # We've reached a dead end
                return self._junct_from_dead_end_cursor(cursor,direction)
            elif (self.in_tunnel(cursor.top_left)
                  or self.in_tunnel(cursor.bottom_right)):
                return self._junct_from_cursor(cursor,direction)
            else:
                cursor = cursor.move(direction)
            cursor = cursor.move(direction)

    def _junct_from_dead_end_cursor(self,cursor,direction):
        (left_x, up_y),(right_x, down_y) = (cursor.top_left, cursor.bottom_right)
        if direction == Rect.UP:
            junct = Rect(top_left=(left_x+1,up_y+1),
                         width=cursor.width-2,
                         height=cursor.width-2)
        elif direction == Rect.DOWN:
            junct = Rect(top_left=(left_x+1,up_y-1),
                         width=cursor.width-2,
                         height=cursor.width-2)
        elif direction == Rect.LEFT:
            junct = Rect(top_left=(left_x+1,up_y+1),
                         width=cursor.height-2,
                         height=cursor.height-2)
        elif direction == Rect.RIGHT:
            junct = Rect(top_left=(left_x-1,up_y+1),
                         width=cursor.height-2,
                         height=cursor.height-2)
        return junct

    def _junct_from_cursor(self,cursor,direction):
        (left_x, up_y),(right_x, down_y) = (cursor.top_left, cursor.bottom_right)
        if direction == Rect.UP:
            width = cursor.width-2
            height = self._move_cursor_to_wall(cursor,direction)
            junct = Rect(top_left=(left_x+1,up_y-height+1),
                         width=width, height=height)
        elif direction == Rect.DOWN:
            width = cursor.width-2
            height = self._move_cursor_to_wall(cursor,direction)
            junct = Rect(top_left=(left_x+1,up_y),
                         width=width, height=height)
        elif direction == Rect.LEFT:
            height = cursor.height-2
            width = 1+self._move_cursor_to_wall(cursor,direction)
            junct = Rect(top_left=(left_x-width+1,up_y+1),
                         width=width, height=height)
        elif direction == Rect.RIGHT:
            height = cursor.height-2
            width = 1+self._move_cursor_to_wall(cursor,direction)            
            junct = Rect(top_left=(left_x,up_y+1),
                         width=width, height=height)
        return junct

    def _move_cursor_to_wall(self,cursor,direction):
        count = 0
        while (self.in_tunnel(cursor.top_left)
               or self.in_tunnel(cursor.bottom_right)):
            cursor = cursor.move(direction)
            count += 1
        return count
            
    def _get_cursor(self, junct, direction):
        cursor = None
        (left_x,up_y),(right_x,down_y) = (junct.top_left, junct.bottom_right)
        if direction == Rect.UP:
            cursor = Rect(top_left=(left_x-1,up_y-1),
                          width=junct.width+2,
                          height=1)
        elif direction == Rect.DOWN:
            cursor = Rect(top_left=(left_x-1,down_y+1),
                          width=junct.width+2,
                          height=1)
        elif direction == Rect.LEFT:
            cursor = Rect(top_left=(left_x-1,up_y-1),
                          width=1,
                          height=junct.height+2)
        elif direction == Rect.RIGHT:
            cursor = Rect(top_left=(right_x+1,up_y-1),
                          width=1,
                          height=junct.height+2)
        return cursor
            
    def graph_from_img(img,root_junct):
        self._graph = graph = networkx.Graph()
        graph.add_node(root_junct)
        dummy_junct = Rect(root_junct.top_left,
                           root_junct.width,
                           root_junct.height-30)
        graph.add_edge(root_junct,dummy_junct)
        #════════════════════
        frontier = deque([root_junct])
        expanded = set()
        while frontier:
            junct = frontier.popleft()
            if junct in expanded:
                continue
            for direction in self._missing_directions(junct):
                neighbor = self.junct_neighbor(junct,direction)
                if neighbor is not None:
                    graph.add_edge(junct,neighbor)
                    frontier.append(neighbor)
            expanded.add(junct)
        return graph

    def _missing_directions(self,junct):
        existing_directions = set()
        jx,jy = junct.top_left
        for neighbor in self._graph.neighbors(junct):
            # Since they are neighbors, we know that either the X or the Y
            # coordinates of the top-left corners are going to match
            nx, ny = neighbor.top_left
            if jx == nx:
                direction = Rect.UP if jy > ny else Rect.DOWN
            else: # We know jy == ny
                direction = Rect.LEFT if jx > nx else Rect.RIGHT
            existing_directions.add(direction)
        return {Rect.UP,Rect.DOWN,Rect.LEFT,Rect.RIGHT}-existing_directions

    def mark_junct(self,junct):
        draw = ImageDraw.Draw(self.img)
        draw.rectangle((junct.top_left,junct.bottom_right),
                       outline=(157,11,11),
                       fill=(157,11,11))
    
def test():
    maze_img = MazeImage(Image.open("maze1.jpg"))
    junct = Rect((326,242),6,6)
    neighbor = maze_img.junct_neighbor(junct,Rect.UP)
    assert neighbor == Rect((326, 194),6,6)
    
    junct = Rect((398,314),6,6)
    neighbor = maze_img.junct_neighbor(junct,Rect.UP)
    assert neighbor == Rect((398, 290),6,6)
    
    junct = Rect((398, 290),6,6)
    neighbor = maze_img.junct_neighbor(junct,Rect.LEFT)
    assert neighbor == Rect((374, 290),6,6)

    junct = Rect((374,338),6,6)
    neighbor = maze_img.junct_neighbor(junct,Rect.LEFT)
    assert neighbor == Rect((350,338),6,6)
    neighbor = maze_img.junct_neighbor(junct,Rect.RIGHT)
    assert neighbor == Rect((422,338),6,6)

    junct = Rect((613,290),6,6)
    cursor = maze_img._get_cursor(junct,Rect.DOWN)
    neighbor = maze_img.junct_neighbor(junct,Rect.DOWN)
    assert neighbor == Rect((613,314),6,6)

def test():
    maze_img = MazeImage(Image.open("maze1.jpg"))
    junct = Rect((41,51),4,6)
    neighbor = maze_img.junct_neighbor(junct,Rect.UP)

def script0():
    img = Image.open("maze1.jpg")
    visible = (217,234,218)
    # dist = math.dist((255,255,255),visible)
    dist = 5
    for x,y in itertools.product(range(img.width),range(img.height)):
        color = img.getpixel((x,y))
        if (color != (255,255,255) and
            0 < math.dist((255,255,255),color) <= 5):
            img.putpixel((x,y),(157,11,11))
    img.save("maze_script0.tiff")

def script1():
    img = Image.open("maze1.jpg")
    for x,y in itertools.product(range(img.width),range(img.height)):
        color = img.getpixel((x,y))
        if color not in MazeImage.COLORS:            
            distances = [(math.dist(COLOR,color),COLOR) for COLOR in MazeImage.COLORS]
            new_color = min(distances)[1]
            img.putpixel((x,y),new_color)
    img.save("maze_script1.tiff")

def script2():
    img = Image.open("maze1.jpg")
    for x,y in itertools.product(range(img.width),range(img.height)):
        color = img.getpixel((x,y))
        dist = math.dist(color,(255,255,255))
        if math.dist(color,(255,255,255))<=
            img.putpixel((x,y),visible)
    img.save("maze_script2.tiff")
