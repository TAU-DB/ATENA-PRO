import itertools

import zss
import anytree

from functools import partial


def anytree_to_zss_tree(t):
    nodes = {}
    root_name = t.node.name
    for n in anytree.PreOrderIter(t.node):
        nodes[n.name] = zss.Node(n.name)
        if n.parent and n.name != root_name:
            nodes[n.parent.name].addkid(nodes[n.name])
        else:
            root_name = n.name
    return nodes[root_name]


def anytree_to_postorder_zss_tree(t, last_node_name):
    nodes = {}
    root_name = t.node.name
    for n in anytree.PreOrderIter(t.node):
        nodes[n.name] = zss.Node(n.name)
        if n.parent and n.name != root_name:
            nodes[n.parent.name].addkid(nodes[n.name])

    nodes[root_name].label = '9' * 10
    for level, level_nodes in enumerate(anytree.LevelOrderGroupIter(t.node)):
        if level == 0:
            continue
        for offset, n in enumerate(level_nodes):
            nodes[n.name].label = nodes[n.parent.name].label[:level-1] + str(offset) + nodes[n.parent.name].label[level:]

    return nodes[root_name], nodes[last_node_name] if last_node_name is not None else None


# def update_cost(a, b):
#     return 0 if a == b else 1


def simple_included(t1, t2, verbose=False):
    distance, ops = zss.distance(t1, t2, get_children=zss.Node.get_children, insert_cost=lambda n: 0,
                                 remove_cost=lambda n: 1, update_cost=lambda a, b: 0, return_operations=True)
    if verbose:
        print(ops)
        print(distance)
    return distance == 0


def prefix_included(t1, t2, last_node, nodes_left, verbose=False):
    magic = 0.01

    def prefix_update_cost(x, y):
        return 0 if x == y else 1

    def prefix_insert_cost(n, last):
        if last is None:
            return magic

        if n.label > last.label:
            return magic

        return 1

    distance, ops = zss.distance(t2, t1, get_children=zss.Node.get_children,
                                 insert_cost=partial(prefix_insert_cost, last=last_node),
                                 remove_cost=lambda n: 0, update_cost=prefix_update_cost,
                                 return_operations=True)
    if verbose:
        print(ops)
        print(distance)
    return distance <= magic * nodes_left


def encode_succint_recursive(n, encoded):  # type: (anytree.Node, list) -> None
    encoded.append('0')
    for c in n.children:
        encode_succint_recursive(c, encoded)
    encoded.append('1')


def encode_succint(t, length=None):
    encoded = []
    encode_succint_recursive(t.node, encoded)
    if length is not None:
        assert len(encoded) <= length, f'Encoded session tree must be shorted than {length}'
        encoded = encoded + [0] * (length - len(encoded))
    return encoded


def encode_to_matrix(t, length):
    encoded = [[0] * length for _ in range(length)]
    root = t.node
    for i, n in enumerate(anytree.PreOrderIter(root)):
        n.n_index = i

    for n in anytree.PreOrderIter(root):
        current_array = encoded[n.n_index]
        if n.parent is not None:
            current_array[n.parent.n_index] = 1
        for c in n.children:
            current_array[c.n_index] = 1

    return list(itertools.chain(*encoded))


def get_tree_info(t, curr_node):  # type: (anytree.RenderTree, anytree.Node) -> list
    root = t.node  # type: anytree.Node
    total_size = len(root.descendants) + 1
    return [total_size, curr_node.depth, curr_node.height, len(curr_node.siblings)]


if __name__ == '__main__':
    a1 = anytree.Node('0')
    a2 = anytree.Node('a', parent=a1)
    a3 = anytree.Node('b', parent=a1)
    a4 = anytree.Node('c', parent=a1)
    a5 = anytree.Node('d', parent=a3)
    a = anytree_to_zss_tree(anytree.RenderTree(a1))

    b1 = anytree.Node('0')
    b2 = anytree.Node('a', parent=b1)
    b3 = anytree.Node('aa', parent=b2)
    b7 = anytree.Node('aaa', parent=b3)
    b4 = anytree.Node('aaaa', parent=b7)
    b5 = anytree.Node('aaab', parent=b7)
    b = anytree_to_zss_tree(anytree.RenderTree(b1))

    min_zss_sub_tree = (zss.Node('9999999999')
                        .addkid(zss.Node('0999999999'))
                        .addkid(zss.Node('1999999999'))
                        .addkid(zss.Node('2999999999'))
                        )

    my_tree = (zss.Node('9999999999')
               .addkid(zss.Node('0999999999'))
               )

    # encoded_tree = []
    # encode_succint_recursive(a1, encoded_tree)
    # print(''.join(encoded_tree).replace('0', '(').replace('1', ')'))

    print(prefix_included(min_zss_sub_tree, my_tree, zss.Node('9999999999'), 100))

