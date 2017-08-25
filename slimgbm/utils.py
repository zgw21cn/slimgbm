from PIL import Image, ImageDraw

def get_tree_depth(tree):
    if tree == None:
        return 0
    leftDepth = get_tree_depth(tree.left_child)
    rightDepth = get_tree_depth(tree.right_child)
    if leftDepth > rightDepth:
        return leftDepth + 1
    if rightDepth >= leftDepth:
        return rightDepth + 1

def draw_tree(tree,file_name="tree.png"):

    h = get_tree_depth(tree) * 100 + 120
    w =get_tree_depth(tree)*100

    img = Image.new('RGB', (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw_node(tree,draw, w / 2, 20)
    img.save(file_name, 'PNG')

def draw_node(tree,draw,x,y):
    if tree.is_leaf:
        draw.ellipse(((x-10,y-10),(x+10,y+10)),outline = "blue")
    else:
        draw.text((x - 30 , y-10),str(tree.feature) , (0, 0, 0))
        draw.text((x - 30 , y),str(tree.threshold) , (0, 0, 0))
        draw.rectangle(((x-30,y-10),(x+30,y+10)),outline = "blue")

    all_width=0
    if tree.left_child:
        all_width+=100
    if tree.right_child:
        all_width+=100

    left=x-all_width/2
    if tree.left_child:
        draw.line((x,y,left+100/2,y+100),fill=(255,0,0))
        draw_node(tree.left_child,draw,left+100/2,y+100)

    if tree.right_child:
        left+=100
        draw.line((x,y,left+100/2,y+100),fill=(255,0,0))
        draw_node(tree.right_child,draw,left+100/2,y+100)




def drawnode(self, draw, x, y):
    if self.type == "function":
      allwidth = 0
      for c in self.children:
        allwidth += c.getwidth()*100
      left = x - allwidth / 2
      #draw the function name
      draw.text((x - 10, y - 10), self.funwrap.name, (0, 0, 0))
      #draw the children
      for c in self.children:
        wide = c.getwidth()*100
        draw.line((x, y, left + wide / 2, y + 100), fill=(255, 0, 0))
        c.drawnode(draw, left + wide / 2, y + 100)
        left = left + wide
    elif self.type == "variable":
      draw.text((x - 5 , y), self.variable.name, (0, 0, 0))
    elif self.type == "constant":
      draw.text((x - 5 , y), self.const.name, (0, 0, 0))
