from treelib import Node, Tree

tree = Tree()
tree.create_node("Harry", "harry")  # root node
tree.create_node("Jane", "jane", parent="harry")
tree.create_node("Bill", "bill", parent="harry")
tree.create_node("Diane", "diane", parent="jane")
tree.create_node("Mary", "mary", parent="diane")
tree.create_node("Mark", "mark", parent="jane")
tree.show()
# print(tree.nodes())
for id, n in tree.nodes.items():
    if n.is_leaf():
        print(id, n)

#
#
# sub_t = tree.subtree('diane')
# sub_t.show()
#
#
# new_tree = Tree()
# new_tree.create_node("n1", 1)  # root node
# new_tree.create_node("n2", 2, parent=1)
# new_tree.create_node("n3", 3, parent=1)
# tree.paste('bill', new_tree)
# tree.show()
#
# tree.remove_node(1)
# tree.show()
#
# tree.move_node('mary', 'harry')
# tree.show()
#
# tree.depth()
#
# node = tree.get_node("bill")
# tree.depth(node)
#
#
# assert False
#
#
# tree.show(line_type="ascii-em")


# class Flower(object):
#     def __init__(self, color):
#         self.color = color

# class FlowerNode(Node):
#     def __init__(self, color):
#         self.color = color

# ftree = Tree()
# ftree.create_node("Root", "root", data=Flower("black"))
# ftree.create_node("F1", "f1", parent='root', data=Flower("white"))
# ftree.create_node("F2", "f2", parent='root', data=Flower("red"))
#
# node = ftree.get_node("root")
# print(node.tag)
# print(node.fpointer)
# print(type(node.data))
# cNode = ftree.get_node("f1")
# print(cNode.bpointer)
#
#
# ftree.show(data_property="color")

