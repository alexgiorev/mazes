import itertools
import math
import networkx
import pdb

from PIL import Image, ImageDraw
from collections import deque

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
    TUNNEL = (255,255,255)
    WALL = (136,170,136)
    CIRCLE_BORDER = (0,0,255)
    CIRCLE_INTERNALS = (255,215,0)
    COLORS = (TUNNEL, WALL, CIRCLE_BORDER, CIRCLE_INTERNALS)
    VERTICAL = True
    HORIZONTAL = False
    
    def __init__(self,img):
        self.img = img
        self.draw = ImageDraw.Draw(img)
        self.PROBE_LENGTH = 10
    
    def in_tunnel(self,xy):
        return self.closest_color(xy) == self.TUNNEL
    
    def closest_color(self,xy):
        def dist(color1,color2):
            # manhattan length
            return (abs(color1[0]-color2[0]) +
                    abs(color1[1]-color2[1]) +
                    abs(color1[2]-color2[2]))
        color = self.img.getpixel(xy)
        distances = [(dist(COLOR,color),COLOR)
                     for COLOR in MazeImage.COLORS]
        return min(distances)[1]

    def junct_neighbor(self, junct, direction):
        junct = junct.move(direction, amount=self.PROBE_LENGTH)
        if not self.in_tunnel(junct.top_left):
            return None
        orientation = (self.VERTICAL if direction in (Rect.LEFT, Rect.RIGHT)
                       else self.HORIZONTAL)
        while True:
            if not self.in_tunnel(junct.top_left):
                # We've reached a dead end without finding a turn. Move back a
                # bit and return the dead end junction
                return junct.move(direction, -3)
            else:
                tunnels = self._probe(junct,orientation)
                if tunnels != (None, None):
                    tunnel = tunnels[0] or tunnels[1]
                    jx,jy = junct.top_left
                    mx,my = self._tunnel_midpoint(tunnel,orientation)
                    if orientation == self.VERTICAL:
                        return Rect(top_left=(mx,jy), width=1, height=1)
                    else:
                        return Rect(top_left=(jx,my), width=1, height=1)
                else:
                    junct = junct.move(direction)

    def _probe(self,junct,orientation):
        """Returns a pair of the pixel coordinates of the tunnels"""
        directions = ((Rect.UP,Rect.DOWN) if orientation == self.VERTICAL
                      else (Rect.LEFT,Rect.RIGHT))
        tunnels = []
        for direction in directions:
            point = junct
            for _ in range(self.PROBE_LENGTH):
                point = point.move(direction)
                if not self.in_tunnel(point.top_left):
                    tunnels.append(None)
                    break
            else:
                tunnels.append(point.top_left)
        return tuple(tunnels)

    def _tunnel_midpoint(self,tunnel_xy,orientation):
        directions = ((Rect.LEFT,Rect.RIGHT) if orientation == self.VERTICAL
                      else (Rect.UP,Rect.DOWN))
        end_points = []
        for direction in directions:
            point = Rect(tunnel_xy,1,1)
            while self.in_tunnel(point.top_left):
                point = point.move(direction)
            end_points.append(point.top_left)
        (x1,y1),(x2,y2) = end_points
        return (math.ceil((x1+x2)/2), math.ceil((y1+y2)/2))
            
    def graph(self, root_junct):
        self._graph = graph = networkx.Graph()
        graph.add_node(root_junct)
        dummy_junct = Rect(top_left=(root_junct.top_left[0],
                                     root_junct.top_left[1]-30),
                           width=1, height=1)
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

#════════════════════════════════════════
# testing
    
def test_neighbors():
    maze_img = MazeImage(Image.open("maze1.jpg"))
    junct = Rect((329,150),1,1)
    left_neighbor = maze_img.junct_neighbor(junct,Rect.LEFT)
    down_neighbor = maze_img.junct_neighbor(junct,Rect.DOWN)
    assert left_neighbor == Rect((281, 150),1,1)
    assert down_neighbor == Rect((329, 173),1,1)
    
    junct = Rect((354, 166),1,1)
    left_neighbor = maze_img.junct_neighbor(junct,Rect.LEFT)
    right_neighbor = maze_img.junct_neighbor(junct,Rect.RIGHT)
    up_neighbor = maze_img.junct_neighbor(junct,Rect.UP)
    assert left_neighbor is right_neighbor is None

    junct = Rect((44,54),1,1)
    down = maze_img.junct_neighbor(junct,Rect.DOWN)
    assert down == Rect((44, 102),1,1)
    right = maze_img.junct_neighbor(down,Rect.RIGHT)
    assert right == Rect((114, 102),1,1)
    up = maze_img.junct_neighbor(right,Rect.UP)
    assert up == Rect((114, 54),1,1)
    left = maze_img.junct_neighbor(up,Rect.LEFT)
    assert left == Rect((90, 54),1,1)
    down = maze_img.junct_neighbor(left,Rect.DOWN)
    assert down == Rect((90, 78),1,1)
    left = maze_img.junct_neighbor(down,Rect.LEFT)
    assert left == Rect((68, 78),1,1)

def draw_graph():
    maze_img = MazeImage(Image.open("maze1.jpg"))
    root_junct = Rect((43,53),1,1)
    graph = maze_img.graph(root_junct)
    for junct1, junct2 in graph.edges:
        xy1, xy2 = junct1.top_left, junct2.top_left
        maze_img.draw.line((xy1,xy2),fill=(0,0,0),width=1)
    maze_img.img.save("maze1_graph.tiff")
    
#════════════════════════════════════════
# misc scripts
    
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

def manhattan_length(color1,color2):
    return (abs(color1[0]-color2[0]) +
            abs(color1[1]-color2[1]) +
            abs(color1[2]-color2[2]))
    
def script3():
    img = Image.open("maze1.jpg")
    for x,y in itertools.product(range(img.width),range(img.height)):
        color = img.getpixel((x,y))
        if color not in MazeImage.COLORS:            
            distances = [(manhattan_length(COLOR,color),COLOR) for COLOR in MazeImage.COLORS]
            new_color = min(distances)[1]
            img.putpixel((x,y),new_color)
    img.save("maze1_manhattan.tiff")
