import itertools
import math
import networkx as nx
import pdb
import os
import shutil

from PIL import Image, ImageDraw, ImageColor
get_color = ImageColor.getrgb
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
    GOAL = (0,0,255)
    COLORS = (TUNNEL, WALL, CIRCLE_BORDER, CIRCLE_INTERNALS)
    VERTICAL = True
    HORIZONTAL = False
    
    def __init__(self,img):
        self.img = img
        self.draw = ImageDraw.Draw(img)
        self.PROBE_LENGTH = 10
    
    def in_tunnel(self,xy):
        return self.closest_color(xy) == self.TUNNEL

    def in_goal(self,xy):
        return self.closest_color(xy) == self.GOAL
    
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
            return None, False
        orientation = (self.VERTICAL if direction in (Rect.LEFT, Rect.RIGHT)
                       else self.HORIZONTAL)
        while True:
            if not self.in_tunnel(junct.top_left):
                goal_junct = self._check_goal(junct.top_left)
                if goal_junct:
                    return goal_junct, True
                else:
                    return junct.move(direction, -3), False
            else:
                tunnels = self._probe(junct,orientation)
                if tunnels != (None, None):
                    tunnel = tunnels[0] or tunnels[1]
                    jx,jy = junct.top_left
                    mx,my = self._tunnel_midpoint(tunnel,orientation)
                    if orientation == self.VERTICAL:
                        return Rect(top_left=(mx,jy), width=1, height=1), False
                    else:
                        return Rect(top_left=(jx,my), width=1, height=1), False
                else:
                    junct = junct.move(direction)

    def _check_goal(self, xy):
        directions = (Rect.DOWN, Rect.RIGHT)
        for direction in directions:
            point = Rect(xy,1,1)
            for _ in range(self.PROBE_LENGTH):
                if self.in_goal(point.top_left):
                    return point
                point = point.move(direction)
        return None
            
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
        self._graph = graph = nx.Graph()
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
                neighbor, is_goal = self.junct_neighbor(junct,direction)
                if neighbor is not None:
                    graph.add_edge(junct,neighbor)
                    graph.nodes[neighbor]["is_goal"] = is_goal
                    if is_goal: expanded.add(neighbor)
                    else: frontier.append(neighbor)
            expanded.add(junct)
        graph.remove_node(dummy_junct)
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

#════════════════════════════════════════
# drawing

def draw_graph(graph,draw,color=get_color("black")):
    for node in graph.nodes:
        draw_node(node,draw,color=color)
    for junct1, junct2 in graph.edges:
        draw_line(junct1,junct2,draw,color=color)

def draw_node(node, draw, color=get_color("black")):
    x,y = node.top_left
    x1,y1 = x-1,y-1
    x2,y2 = x1+2,y1+2
    draw.rectangle(((x1,y1),(x2,y2)),fill=color,outline=color)

def draw_line(node1, node2, draw, color=get_color("black")):
    xy1, xy2 = node1.top_left, node2.top_left
    draw.line((xy1,xy2),fill=color,width=1)
    
def draw_graph_script():
    img = Image.open("maze1_no_big.jpg")
    maze_img = MazeImage(img)
    root_junct = Rect((43,53),1,1)
    graph = maze_img.graph(root_junct)
    blank = Image.new("RGB", maze_img.img.size, MazeImage.TUNNEL)
    draw = ImageDraw.Draw(blank)
    draw_graph(graph,draw)
    img.save("draw_graph_result.tiff")    
    return graph

def draw_path(nodes,draw,color=(0,0,0)):
    for node in nodes:
        draw_node(node,draw)
    iter1, iter2 = iter(nodes), iter(nodes)
    try: next(iter2)
    except StopIteration: pass
    for node1,node2 in zip(iter1,iter2):
        draw_line(node1, node2, draw, color)

#════════════════════════════════════════
# search algorithms

class Searcher:
    def __init__(self,graph,root):
        self.graph = graph
        self.root = root

    #════════════════════════════════════════
    # breadth-first-search
    
    def bfs(self):
        root, graph = self.root, self.graph
        self.bfs_search_tree = search_tree = nx.DiGraph()
        search_tree.add_node(root)
        self.bfs_frontier = frontier = deque([root])
        self.bfs_explored = explored = set()
        self.bfs_did_init()
        while frontier:
            junct = frontier.popleft()
            if junct in explored:
                continue
            self.bfs_current = junct
            self.bfs_before_iteration()            
            for neighbor in graph.neighbors(junct):
                if neighbor in explored:
                    continue
                elif neighbor in search_tree:
                    # NEIGHBOR is in FRONTIER
                    continue
                elif graph.nodes[neighbor].get("is_goal"):
                    path = [neighbor]
                    while junct is not None:
                        path.append(junct)
                        junct = next(search_tree.predecessors(junct),None)
                    path.reverse()
                    self.bfs_end(path)
                    return path, search_tree
                else:
                    search_tree.add_edge(junct, neighbor)
                    self.bfs_did_extend_search_tree((junct,neighbor))
                    frontier.append(neighbor)
            explored.add(junct)
        self.bfs_end(None)
        return None, search_tree

    def bfs_did_init(self):
        self.bfs_DIR = DIR = "maze1_bfs"
        try:
            shutil.rmtree(DIR)
        except FileNotFoundError:
            pass
        os.mkdir(DIR)
        self.bfs_iter_count = 0
        self.bfs_canvas = Image.open("maze1_no_big.jpg")
        self.bfs_draw = ImageDraw.Draw(self.bfs_canvas)

    def bfs_before_iteration(self):
        pass

    def bfs_did_extend_search_tree(self, edge):
        node1, node2 = edge
        draw_node(node1, self.bfs_draw)
        draw_node(node2, self.bfs_draw)
        draw_line(node1, node2, self.bfs_draw)
        img_name = self.bfs_next_img_name()
        self.bfs_canvas.save(os.path.join(self.bfs_DIR, img_name))
        
    def bfs_next_img_name(self):
        img_name = f"{str(self.bfs_iter_count).rjust(6,'0')}.jpg"
        self.bfs_iter_count += 1
        return img_name
        
    def bfs_end(self, path):
        if path is not None:
            draw_graph(self.bfs_search_tree, self.bfs_draw, color=get_color("gray"))
            draw_path(path, self.bfs_draw)
            img_name = self.bfs_next_img_name()
            self.bfs_canvas.save(os.path.join(self.bfs_DIR, img_name))
    
def search_bfs_script0():
    img = Image.open("maze1_no_big.jpg")
    maze_img = MazeImage(img)
    root_junct = Rect((43,53),1,1)
    graph = maze_img.graph(root_junct)
    searcher = Searcher(graph, root_junct)
    path, search_tree = searcher.bfs()
    # draw = ImageDraw.Draw(img)
    # draw_graph(search_tree,draw,color=get_color("gray"))
    # draw_path(path,draw,color=get_color("black"))
    # img.save("maze1_path.tiff")
    # return graph

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

def graph_dot(graph):
    def node_name(node):
        return str(node.top_left)
    lines = ["graph {"]
    for node,attrs in graph.nodes.items():
        lines.append(f'    "{node_name(node)}";')
    for (node1,node2),attrs in graph.edges.items():
        name1, name2 = node_name(node1), node_name(node2)
        lines.append(f'    "{name1}"--"{name2}";')
    lines.append("}"); lines.append("")
    return "\n".join(lines)
