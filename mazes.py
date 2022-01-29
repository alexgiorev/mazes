import itertools
import math
import networkx as nx
import pdb
import os
import shutil

from PIL import Image, ImageDraw, ImageColor
COLOR = ImageColor.getrgb
from collections import deque

class Rect:
    LEFT = (-1,0)
    RIGHT = (1,0)
    UP = (0,-1)
    DOWN = (0,1)
    DIRECTIONS = (LEFT,RIGHT,UP,DOWN)
    
    def __init__(self, top_left,width=None,height=None,bottom_right=None):
        self.top_left = top_left
        if bottom_right is not None:
            self.width = bottom_right[0]-top_left[0]+1
            self.height = bottom_right[1]-top_left[1]+1
        else:
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

    @property
    def lx(self):
        return self.top_left[0]
    @property
    def rx(self):
        return self.top_left[0]+self.width-1
    @property
    def uy(self):
        return self.top_left[1]
    @property
    def dy(self):
        return self.top_left[1]+self.height-1
    
    @property
    def xs(self):
        x = self.top_left[0]
        return range(x, x+self.width)

    @property
    def ys(self):
        y = self.top_left[1]
        return range(y, y+self.height)

    @property
    def xys(self):
        for y in self.ys:
            for x in self.xs:
                yield (x,y)

    def __contains__(self,xy):
        lx,uy = self.top_left
        rx,dy = self.bottom_right
        return (lx <= xy[0] <= rx) and (uy <= xy[1] <= dy)

class MazeImage:
    TUNNEL = (255,255,255)
    WALL = (136,170,136)
    GOAL = (0,0,255)    
    CIRCLE_BORDER = (0,0,255)
    CIRCLE_INTERNALS = (255,215,0)
    GOAL = (0,0,255)
    COLORS = (TUNNEL, WALL, GOAL, CIRCLE_BORDER, CIRCLE_INTERNALS)
    VERTICAL = True
    HORIZONTAL = False
    TUNNEL_SIZE = 7
    PROBE_LENGTH = 2*TUNNEL_SIZE
    
    def __init__(self,img):
        self.img = img
        self.draw = ImageDraw.Draw(img)

    # utils
    #════════════════════════════════════════
    
    def in_tunnel(self,xy):
        return self.closest_color(xy) == self.TUNNEL
    
    def rect_in_tunnel(self,rect):
        for xy in rect.xys:
            if not self.in_tunnel(xy):
                return False
        return True
    
    def in_wall(self,xy):
        return self.closest_color(xy) == self.WALL
    
    def rect_in_wall(self,rect):
        for xy in rect.xys:
            if not self.in_wall(xy):
                return False
        return True
    
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

    @property
    def xys(self):
        for y in range(self.img.height):
            for x in range(self.img.width):
                yield x,y

    def next_color(self, xy, direction,inside=True):
        """INSIDE controls whether to leave point inside the next color or right before it"""
        x,y = xy
        dx,dy = direction
        color = self.closest_color(xy)
        while True:
            x+=dx; y+=dy
            try:
                new_color = self.closest_color((x,y))
            except IndexError:
                return None
            if new_color != color:
                if not inside: x-=dx; y-=dy
                return x,y

    def fill_rect(self,rect,color):
        self.draw.rectangle((rect.top_left,rect.bottom_right),
                            fill=color,
                            outline=color)

    @property
    def maze_top_left(self):
        if hasattr(self,"_maze_rect"):
            return self._maze_rect.top_left
        else:
            for y in range(30):
                for x in range(30):
                    if self.in_wall((x,y)):
                        return (x,y)
        
    @property
    def maze_rect(self):
        if hasattr(self, "_maze_rect"):
            return self._maze_rect
        else:
            img_width, img_height = self.img.size
            top_left = self.maze_top_left
            right_x, bottom_y = None, None
            try:
                for y in range(img_height-1,img_height-31,-1):
                    for x in range(img_width-1,img_width-31,-1):
                        if not self.in_tunnel((x,y)):
                            right_x, bottom_y = x,y
                            raise StopIteration
            except StopIteration: pass
            width = right_x - top_left[0] + 1
            height = bottom_y - top_left[1] + 1
            self._maze_rect = Rect(top_left,width,height)
            return self._maze_rect

    def neighbor_pixels(self, xy, directions=None):
        for xy0 in self.neighbor_xys(xy,directions):
            yield self.closest_color(xy0)

    def neighbor_xys(self, xy, directions=None):
        if directions is None:
            directions = ((1,0),(-1,0),(0,1),(0,-1),
                          (1,1),(1,-1),(-1,1),(-1,-1))
        elif directions == "straight":
            directions = ((1,0),(-1,0),(0,1),(0,-1))
        elif directions == "diagonal":
            directions = ((1,1),(1,-1),(-1,1),(-1,-1))
        else: # directions is an iterable
            pass
        x,y = xy
        for dx,dy in directions:
            x0,y0 = x+dx,y+dy
            if 0 <= x0 < self.img.width and 0 <= y0 < self.img.height:
                yield x0,y0

    def foreach(self,xy,func,
                child_pred=lambda xy:True,
                directions="straight"):
        pending = deque([xy])
        visited = set()
        try:
            while pending:
                xy = pending.popleft()
                if xy in visited:
                    continue
                func(xy)
                visited.add(xy)
                pending.extend((nxy for nxy in self.neighbor_xys(xy, directions)
                                if child_pred(nxy) and nxy not in visited))
        except StopIteration:
            pass

    def map_manhattan(self):
        img = self.img
        for xy in self.xys:
            img.putpixel(xy,self.closest_color(xy))

    # process_image
    #════════════════════════════════════════
    def process_image(self):
        img = self.img
        for xy in self.xys:
            img.putpixel(xy,self.closest_color(xy))
        self.remove_initial_circle()
        self.remove_goal_circle()
        self.fill_big_tunnels()
        self.fix_tunnels()

    # process_image.fill_big_tunnels
    #════════════════════════════════════════
    
    def fill_big_tunnels(self):
        maze_rect = self.maze_rect
        tunnel_size = self.TUNNEL_SIZE
        for y in maze_rect.ys:
            for x in maze_rect.xs:
                if self._on_top_left_of_big_tunnel(x,y):
                    width, height = self._big_tunnel_dimensions(x,y)
                    fill_rect = Rect((x+tunnel_size, y+tunnel_size),
                                     width-2*tunnel_size,
                                     height-2*tunnel_size)
                    self.draw.rectangle((fill_rect.top_left,
                                         fill_rect.bottom_right),
                                        fill=self.WALL)

    def _on_top_left_of_big_tunnel(self, x,y):
        try:
            for d in range(1,2*self.TUNNEL_SIZE+1):
                to_check = [(x+d,y+d),(x+d,y),(x,y+d)]
                if not all(map(self.in_tunnel, to_check)):
                    return False
        except IndexError:
            return False
        return True

    def _big_tunnel_dimensions(self, x,y):                        
        height, width = 2*self.TUNNEL_SIZE, 2*self.TUNNEL_SIZE
        # compute the height
        try:
            for height in range(height+1, self.img.height):
                y0 = y+height
                for x0 in range(x,x+width):
                    if not self.in_tunnel((x0,y0)):
                        raise StopIteration
        except StopIteration: pass
        # compute the width
        try:
            for width in range(width+1,self.img.width):
                x0 = x+width
                # start with (y+1) so that bumps don't deceive us into ending the loop prematurely
                for y0 in range(y+1,y+height):
                    if not self.in_tunnel((x0,y0)):
                        raise StopIteration
        except StopIteration: pass
        return width, height
    
    # process_image.fix_tunnels
    #════════════════════════════════════════
    
    def fix_tunnels(self):
        horizontal_xys = set()
        vertical_xys = set()
        for xy in self.maze_rect.xys:
            if xy not in horizontal_xys and self._check_horizontal(xy):
                rect = self._fix_horizontal(xy)
                horizontal_xys.update(rect.xys)
            if xy not in vertical_xys and self._check_vertical(xy):
                rect = self._fix_vertical(xy)
                vertical_xys.update(rect.xys)

    def _check_horizontal(self,xy):
        x,y = xy
        for x0 in range(x,x+self.PROBE_LENGTH):
            if not self.in_tunnel((x0,y)):
                return False
        return True

    def _check_vertical(self,xy):
        x,y = xy
        for y0 in range(y,y+self.PROBE_LENGTH):
            if not self.in_tunnel((x,y0)):
                return False
        return True

    def _tunnels_at(self,xy):
        result = []
        for dx,dy in [(1,0),(0,1)]:
            x,y = xy
            for _ in range(self.PROBE_LENGTH):
                if not self.in_tunnel((x,y)):
                    result.append(False)
                    break
                else:
                    x+=dx; y+=dy
            else:
                result.append(True)
        return result
                
    def _fix_horizontal(self,xy):
        """Fixes the horizontal tunnel at XY and returns its Rect"""
        x,y = xy
        size = self.TUNNEL_SIZE
        # find the X of the right wall
        right_wall_x = x+self.PROBE_LENGTH
        while True:
            probe = Rect((right_wall_x,y),width=1,height=size)
            if self.rect_in_wall(probe):
                break
            else:
                right_wall_x += 1
        rect = Rect(xy, width=right_wall_x-x, height=size)
        for x0 in rect.xs:
            tunnel_brush = Rect((x0,y),width=1,height=size)
            self.fill_rect(tunnel_brush,self.TUNNEL)
            wall_brushes = [Rect((x0,y-size),width=1,height=size),
                            Rect((x0,y+size),width=1,height=size)]
            for wall_brush in wall_brushes:
                if not self.rect_in_tunnel(wall_brush):
                    self.fill_rect(wall_brush, self.WALL)
        return rect
            
    def _fix_vertical(self,xy):
        """Fixes the vertical tunnel at XY and returns its Rect"""
        x,y = xy
        size = self.TUNNEL_SIZE
        # find the y of the bottom wall
        bottom_wall_y = y+self.PROBE_LENGTH
        while True:
            probe = Rect((x,bottom_wall_y),width=size,height=1)
            if self.rect_in_wall(probe):
                break
            else:
                bottom_wall_y += 1
        rect = Rect(xy, width=size, height=bottom_wall_y-y)
        for y0 in rect.ys:
            tunnel_brush = Rect((x,y0),width=size,height=1)
            self.fill_rect(tunnel_brush,self.TUNNEL)
            wall_brushes = [Rect((x-size,y0),width=size,height=1),
                            Rect((x+size,y0),width=size,height=1)]
            for wall_brush in wall_brushes:
                if not self.rect_in_tunnel(wall_brush):
                    self.fill_rect(wall_brush, self.WALL)
        return rect

    # process_image.remove_circles
    #════════════════════════════════════════
    
    def remove_initial_circle(self):
        # utility functions
        #════════════════════
        def has_yellow(rect):
            for xy in rect.xys:
                if self.closest_color(xy) == self.CIRCLE_INTERNALS:
                    return True
            return False
        def top_strip():
            point = mx,ty-3
            lx,ly = self.next_color(point, Rect.LEFT)
            rx,ry = self.next_color(point, Rect.RIGHT)
            width = rx-lx-1
            x = lx+1; y = self.maze_rect.top_left[1]
            return Rect((x,y),width,1)
        def right_strip():
            point = bx+3,my
            if self.in_wall(point):
                return None
            ux,uy = self.next_color(point, Rect.UP)
            dx,dy = self.next_color(point, Rect.DOWN)
            rect = Rect((ux,uy+1),width=1,height=dy-uy-1)
            while has_yellow(rect):
                rect = rect.move(Rect.RIGHT)
            return rect
        def bottom_strip():
            point = mx,by+3
            if self.in_wall(point):
                return None
            lx,ly = self.next_color(point, Rect.LEFT)
            rx,ry = self.next_color(point, Rect.RIGHT)
            rect = Rect((lx+1,ly),width=rx-lx-1,height=1)
            while has_yellow(rect):
                rect = rect.move(Rect.DOWN)
            return rect
        # remove the circle
        #════════════════════
        circle_rect = self._circle_rect()
        tx,ty = circle_rect.top_left
        mx,my = circle_rect.midpoint
        bx,by = circle_rect.bottom_right
        top = top_strip()
        bottom = bottom_strip()
        right = right_strip()
        rects = []
        if bottom:
            rects.append(Rect(top.top_left,
                              width=top.width,
                              height=bottom.dy-top.uy+1))
        if right:
            rects.append(Rect((top.lx,right.uy),
                              width=right.rx-top.lx+1,
                              height=right.height))
        for rect in rects:
            self.fill_rect(rect,self.TUNNEL)
        img = self.img
        for xy in circle_rect.xys:
             if not any((xy in rect for rect in rects)):
                 img.putpixel(xy,self.WALL)
        # fill the opening
        #════════════════════
        point0 = self.maze_rect.top_left
        point1 = self.next_color(point0,Rect.RIGHT)
        point2 = self.next_color(point1,Rect.RIGHT)
        point3 = self.next_color(point2,Rect.DOWN)
        rect = Rect(top_left=point0,
                    bottom_right=(point3[0],point3[1]-1))
        self.fill_rect(rect,self.WALL)
            
    def _circle_rect(self):
        # find the first tunnel
        x,y = self.maze_top_left
        x0,y0 = self.next_color((x,y),Rect.RIGHT)
        x1,y1 = self.next_color((x0,y0),Rect.RIGHT)
        x0 += (x1-x0)//2 # move to the middle of the tunnel
        # move down to the top border of the circle
        while not self.closest_color((x0,y0)) == self.CIRCLE_BORDER:
            x0,y0 = self.next_color((x0,y0),(0,1))
        top_y = y0
        # move down to the bottom border of the circle
        x0,y0 = self.next_color((x0,y0),(0,1))
        while not self.closest_color((x0,y0)) == self.CIRCLE_BORDER:
            x0,y0 = self.next_color((x0,y0),(0,1))
        x0,y0 = self.next_color((x0,y0),(0,1))
        bottom_y = y0-1
        # go back to the center
        center = x0,math.ceil((top_y+bottom_y)/2)
        x0,y0 = center
        # go to the left border
        while not self.closest_color((x0,y0)) == self.CIRCLE_BORDER:
            x0,y0 = self.next_color((x0,y0),(-1,0))
        x0,y0 = self.next_color((x0,y0),(-1,0))
        left_x = x0+1
        # go to the right border
        x0,y0 = center
        while not self.closest_color((x0,y0)) == self.CIRCLE_BORDER:
            x0,y0 = self.next_color((x0,y0),(1,0))
        x0,y0 = self.next_color((x0,y0),(1,0))
        right_x = x0-1
        return Rect((left_x,top_y),
                    right_x-left_x+1,
                    bottom_y-top_y+1)

    def remove_goal_circle(self):
        # remove the circle
        #════════════════════
        def child_pred(cxy):
            return (self.closest_color(cxy) == self.GOAL
                    or is_floater(cxy))
        def is_floater(xy):
            if not self.in_wall(xy):
                return False
            x,y = xy
            up,down,left,right = (x,y-1),(x,y+1),(x-1,y),(x+1,y)
            return ((not self.in_wall(up) and not self.in_wall(down)) or
                    (not self.in_wall(left) and not self.in_wall(right)))
        img = self.img        
        def func(xy):
            img.putpixel(xy,self.TUNNEL)
        point0 = self.maze_rect.bottom_right
        point1 = self.next_color(point0, Rect.LEFT)
        point2 = self.next_color(point1, Rect.UP)
        self.foreach(point2,func,child_pred)
        # fill the gap
        #════════════════════
        point2 = self.next_color(point1, Rect.LEFT)
        point3 = self.next_color(point2, Rect.UP)
        top_left = (point3[0], point3[1]+1)
        rect = Rect(top_left, bottom_right=point0)
        self.fill_rect(rect, self.WALL)
                    
    # Graph creation
    #════════════════════════════════════════

    def graph(self, trace=False):
        if trace:
            trace_img = self.img.copy()
            trace_draw = ImageDraw.Draw(trace_img)
            trace_index = 0
            trace_images = []
            trace_dir = "graph_trace"
        root_junct = self.root_junct
        goal_junct = self.goal_junct
        self._graph = graph = nx.Graph()
        graph.add_nodes_from([root_junct,
                              (goal_junct,{"is_goal":True})])
        #════════════════════
        frontier = deque([root_junct])
        expanded = set()
        while frontier:
            junct = frontier.popleft()
            if junct in expanded:
                continue
            junct_x, junct_y = junct.top_left
            for direction in self._missing_directions(junct):
                neighbor = self.junct_neighbor(junct,direction)
                if neighbor is not None:
                    neigh_x, neigh_y = neighbor.top_left
                    cost = abs(junct_x-neigh_x)+abs(junct_y-neigh_y)
                    graph.add_edges_from([(junct,neighbor,{"cost":cost})])
                    frontier.append(neighbor)
                    if trace:
                        draw_edge(junct, neighbor, trace_draw)
                        trace_copy = trace_img.copy()
                        trace_copy.info["name"] = str(trace_index).rjust(6,"0")+".tiff"
                        trace_index += 1
                        trace_images.append(trace_copy)
            expanded.add(junct)
        graph.graph["image"] = self.img
        graph.graph["root"] = root_junct
        if trace:
            try: shutil.rmtree(trace_dir)
            except FileNotFoundError: pass
            os.mkdir(trace_dir)
            for image in trace_images:
                image.save(os.path.join(trace_dir,image.info["name"]))
        return graph

    @property
    def root_junct(self):
        point = self.maze_rect.top_left
        point = self.next_color(point,(1,1))
        point = self.next_color(point,Rect.UP,inside=False)
        point = self.next_color(point,Rect.LEFT,inside=False)
        point = (point[0]+self.TUNNEL_SIZE//2,
                 point[1]+self.TUNNEL_SIZE//2)
        return Rect(point,1,1)

    @property
    def goal_junct(self):
        point = self.maze_rect.bottom_right
        point = self.next_color(point,(-1,-1))
        point = self.next_color(point,Rect.DOWN,inside=False)
        point = self.next_color(point,Rect.RIGHT,inside=False)
        point = (point[0]-self.TUNNEL_SIZE//2,
                 point[1]-self.TUNNEL_SIZE//2)
        return Rect(point,1,1)
    
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

    def junct_neighbor(self, junct, direction):
        junct = junct.move(direction, amount=self.PROBE_LENGTH)
        if not self.in_tunnel(junct.top_left):
            return None
        orientation = (self.VERTICAL if direction in (Rect.LEFT, Rect.RIGHT)
                       else self.HORIZONTAL)
        while True:
            if not self.in_tunnel(junct.top_left):
                return junct.move(direction, -(self.TUNNEL_SIZE//2+1))
            else:
                tunnels = self._probe(junct,orientation)
                if tunnels != (None, None):
                    tx,ty = tunnels[0] or tunnels[1]
                    jx,jy = junct.top_left
                    if orientation == self.VERTICAL:
                        return Rect(top_left=(tx,jy), width=1, height=1)
                    else:
                        return Rect(top_left=(jx,ty), width=1, height=1)
                else:
                    junct = junct.move(direction)
            
    def _probe(self,junct,orientation):
        directions, other_directions = (((Rect.UP,Rect.DOWN),(Rect.LEFT,Rect.RIGHT))
                                        if orientation == self.VERTICAL
                                        else ((Rect.LEFT,Rect.RIGHT),(Rect.UP,Rect.DOWN)))
        tunnels = []
        for direction in directions:
            point = junct
            for _ in range(self.PROBE_LENGTH):
                point = point.move(direction)
                if not self.in_tunnel(point.top_left):
                    tunnels.append(None)
                    break
            else:
                point = point.top_left
                point1 = self.next_color(point, other_directions[0])
                point2 = self.next_color(point, other_directions[1])
                tunnels.append(((point1[0]+point2[0])//2,
                                (point1[1]+point2[1])//2))
        return tuple(tunnels)

class Searcher:
    def __init__(self,graph,roots=None):
        self.graph = graph
        self.root = roots or graph.graph["root"]

    # breadth-first-search
    #════════════════════════════════════════
    
    def bfs(self):
        root, graph = self.root, self.graph
        self.search_tree = search_tree = nx.DiGraph()
        search_tree.add_node(root)
        self.frontier = frontier = deque([root])
        self.explored = explored = set()
        self.did_init()
        while frontier:
            junct = frontier.popleft()
            if junct in explored:
                continue
            self.current = junct
            self.before_expansion()            
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
                    self.end(path)
                    return path, search_tree
                else:
                    search_tree.add_edge(junct, neighbor)
                    self.did_extend_search_tree((junct,neighbor))
                    frontier.append(neighbor)
            explored.add(junct)
        self.end(None)
        return None, search_tree

    # hooks
    #════════════════════════════════════════
    
    def did_init(self):
        self.DIR = DIR = "search_result"
        try:
            shutil.rmtree(DIR)
        except FileNotFoundError:
            pass
        os.mkdir(DIR)
        self.iter_count = 0
        self.canvas = self.graph.graph["image"].copy()
        self.draw = ImageDraw.Draw(self.canvas)

    def before_expansion(self):
        pass

    def did_extend_search_tree(self, edge):
        node1, node2 = edge
        draw_edge(node1, node2, self.draw)
        img_name = self._next_img_name()
        self.canvas.save(os.path.join(self.DIR, img_name),quality=100)
        
    def _next_img_name(self):
        img_name = f"{str(self.iter_count).rjust(6,'0')}.jpg"
        self.iter_count += 1
        return img_name
        
    def end(self, path):
        if path is not None:
            draw_graph(self.search_tree, self.draw, color=COLOR("gray"))
            draw_path(path, self.draw)
            img_name = self._next_img_name()
            self.canvas.save(os.path.join(self.DIR, img_name))

    # best_first_search
    #════════════════════════════════════════
    
    def best_first_search(graph,root):
        raise NotImplementedError

# drawing
#════════════════════════════════════════

def draw_graph(graph,draw,color=COLOR("black")):
    for junct1, junct2 in graph.edges:
        draw_edge(junct1,junct2,draw,color=color)

def draw_node(node, draw, color=COLOR("black")):
    x,y = node.top_left
    x1,y1 = x-1,y-1
    x2,y2 = x1+2,y1+2
    draw.rectangle(((x1,y1),(x2,y2)),fill=color,outline=color)

def draw_edge(node1, node2, draw, color=COLOR("black")):
    draw_node(node1, draw, color); draw_node(node2, draw, color)
    xy1, xy2 = node1.top_left, node2.top_left
    draw.line((xy1,xy2),fill=color,width=1)

def draw_path(nodes,draw,color=(0,0,0)):
    iter1, iter2 = iter(nodes), iter(nodes)
    try: next(iter2)
    except StopIteration: pass
    for node1,node2 in zip(iter1,iter2):
        draw_edge(node1, node2, draw, color)

# misc scripts
#════════════════════════════════════════

# search_bfs
def scratch():
    img = Image.open("images/maze4.tiff")
    mimg = MazeImage(img)
    graph = mimg.graph()
    searcher = Searcher(graph)
    path, search_tree = searcher.bfs()
    # draw = ImageDraw.Draw(img)
    # draw_graph(search_tree,draw,color=COLOR("gray"))
    # draw_path(path,draw,color=COLOR("black"))
    # img.save("maze1_path.tiff")
    # return graph

def process_image(fullname):
    path = os.path.join("images",fullname)
    img = Image.open(path)
    MazeImage(img).process_image()
    img.save(path)
