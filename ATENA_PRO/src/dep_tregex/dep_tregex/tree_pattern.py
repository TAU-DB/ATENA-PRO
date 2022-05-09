import itertools
import re

class TreePattern:
    """
    Base class for a tree pattern.
    Tree pattern matches a single node in dependency tree.
    """

    def match(self, tree, node, backrefs_map):
        """
        Return whether a node matches this pattern.

        - tree: a Tree
        - node: node index (1-based, 0 means "root") to match on
        - backrefs_map: contains backreferences to nodes in tree
          (dict: unicode -> int), which will be available after the
          whole-pattern match. Backreferences also can be used to communicate
          with sub-patterns (see e.g. SetBackref and EqualsBackref).

          All patterns should comply with the invariant:

          * If pattern not matches, backrefs_map should be left intact.
          * If pattern matches, it may write something to backrefs_map.
        """
        raise NotImplementedError()

def compile_regex(pattern, ignore_case, anywhere):
    """
    Return Python compiled regex. Match your string against it with
    'r.search(s)'.

    - ignore_case: ignore case of s.
    - anywhere: if False, r.search(s) does a whole-string match.
    """
    flags = re.UNICODE
    if ignore_case:
        flags = flags | re.IGNORECASE
    if not anywhere:
        pattern = '^' + pattern + '$'
    return re.compile(pattern, flags)

## ----------------------------------------------------------------------------
#                                  Children

class HasLeftChild(TreePattern):
    def __init__(self, condition):
        self.condition = condition

    def match(self, tree, node, backrefs_map):
        for child in tree.children(node):
            if child > node:
                continue
            if self.condition.match(tree, child, backrefs_map):
                return True
        return False

class HasRightChild(TreePattern):
    def __init__(self, condition):
        self.condition = condition

    def match(self, tree, node, backrefs_map):
        for child in tree.children(node):
            if child < node:
                continue
            if self.condition.match(tree, child, backrefs_map):
                return True
        return False

class HasChild(TreePattern):
    def __init__(self, condition):
        self.condition = condition

    def match(self, tree, node, backrefs_map):
        for child in tree.children(node):
            if self.condition.match(tree, child, backrefs_map):
                return True
            return False

class HasSuccessor(TreePattern):
    def __init__(self, condition):
        self.condition = condition

    def match(self, tree, node, backrefs_map):
        for child in tree.children_recursive(node):
            if self.condition.match(tree, child, backrefs_map):
                return True
        return False

class HasAdjacentLeftChild(TreePattern):
    def __init__(self, condition):
        self.condition = condition

    def match(self, tree, node, backrefs_map):
        for child in tree.children(node):
            if child + 1 != node:
                continue
            if self.condition.match(tree, child, backrefs_map):
                return True
        return False


class HasAdjacentRightChild(TreePattern):
    def __init__(self, condition):
        self.condition = condition

    def match(self, tree, node, backrefs_map):
        for child in tree.children(node):
            if child - 1 != node:
                continue
            if self.condition.match(tree, child, backrefs_map):
                return True
        return False


class MultiID(object):
    def __init__(self):
        self.conditions = []
        self.allow_more = False

class HasAdjacentRightChildOnly(TreePattern):
    def __init__(self, multi_id):
        self.condition = HasAdjacentRightChildOnlyInner(multi_id)

    def match(self, tree, node, backrefs_map):
        children = tree.children(node)
        return self.condition.match(tree, children, backrefs_map)

    def could(self, tree, node, backrefs_map, position):
        children = tree.children(node)
        return self.condition.could(tree, children, backrefs_map, position, parent=node)

class HasAdjacentRightChildOnlyInner(TreePattern):
    def __init__(self, multi_id):
        self.conditions = multi_id.conditions
        self.allow_more = multi_id.allow_more  # only supports nodes and * at the end

    def match_condition_recursive(self, tree, nodes, backrefs_map, i):
        if i == len(self.conditions):
            yield True
            return

        old_map = backrefs_map.copy()
        for _ in self.conditions[i].match(tree, nodes[i], backrefs_map):
            for _ in self.match_condition_recursive(tree, nodes, backrefs_map, i + 1):
                yield True
            backrefs_map.clear()
            backrefs_map.update(old_map)

    def could_condition_recursive(self, tree, nodes, backrefs_map, position, i):
        if i == len(self.conditions):
            yield 0
            return

        old_map = backrefs_map.copy()
        for my_cost in self.conditions[i].could(tree, nodes[i], backrefs_map, position):
            for other_cost in self.could_condition_recursive(tree, nodes, backrefs_map, position, i + 1):
                yield my_cost + other_cost
            backrefs_map.clear()
            backrefs_map.update(old_map)

    def match(self, tree, nodes, backrefs_map):
        if len(nodes) != len(self.conditions) and not self.allow_more:
            return
        if len(nodes) < len(self.conditions):
            return
        for _ in self.match_condition_recursive(tree, nodes, backrefs_map, 0):
            yield True
        return True

    def could(self, tree, nodes, backrefs_map, position, parent):
        edit_cost = 0
        nodes_to_remove = set()
        if len(nodes) < len(self.conditions) and parent != -1:
            to_add = len(self.conditions) - len(nodes)
            tree.append(['*'] * to_add, [''] * to_add, ['_'] * to_add, ['_'] * to_add, [[] for _ in range(to_add)], [parent] * to_add, ['_'] * to_add)
            edit_cost += to_add
            old_children = set(nodes)
            nodes = tree.children(parent)
            nodes_to_remove = set(nodes).difference(old_children)
            if not tree.is_valid(position):
                tree.delete(nodes_to_remove)
                return
        elif parent == -1:
            nodes = [-1] * len(self.conditions)
            edit_cost = len(nodes)
        elif len(nodes) > len(self.conditions) and not self.allow_more:
            return
        for cost in self.could_condition_recursive(tree, nodes, backrefs_map, position, 0):
            yield edit_cost + cost
        if nodes_to_remove:
            tree.delete(nodes_to_remove)

class HasSuccessorList(TreePattern):
    def __init__(self, multi_id):
        self.condition = HasSuccessorListInner(multi_id)

    def match(self, tree, node, backrefs_map):
        successors = tree.children_recursive(node)
        return self.condition.match(tree, successors, backrefs_map)

    def could(self, tree, node, backrefs_map, position):
        successors = tree.children_recursive(node)
        return self.condition.could(tree, successors, backrefs_map, position, parent=node)

class HasSuccessorListInner(TreePattern):
    def __init__(self, multi_id):
        self.conditions = multi_id.conditions
        self.allow_more = multi_id.allow_more

    def _get_nodes_options(self, nodes, support_new=False):
        options = nodes + ([-1] if support_new else [])
        all_permutations = list(itertools.product(*([options] * len(self.conditions))))
        valid_permutations = filter(lambda x: len(x) == len(set(x)), all_permutations)
        return valid_permutations

    def match_condition_recursive(self, tree, nodes, backrefs_map, i):
        if i == len(self.conditions):
            yield True
            return

        old_map = backrefs_map.copy()
        for _ in self.conditions[i].match(tree, nodes[i], backrefs_map):
            for _ in self.match_condition_recursive(tree, nodes, backrefs_map, i + 1):
                yield True
            backrefs_map.clear()
            backrefs_map.update(old_map)

    def could_condition_recursive(self, tree, nodes, backrefs_map, position, i):
        if i == len(self.conditions):
            yield 0
            return

        old_map = backrefs_map.copy()
        for my_cost in self.conditions[i].could(tree, nodes[i], backrefs_map, position):
            for other_cost in self.could_condition_recursive(tree, nodes, backrefs_map, position, i + 1):
                yield my_cost + other_cost
            backrefs_map.clear()
            backrefs_map.update(old_map)

    def match(self, tree, nodes, backrefs_map):
        if not self.allow_more and len(nodes) != len(self.conditions):
            return
        if len(nodes) < len(self.conditions):
            return
        for nodes_option in self._get_nodes_options(nodes):
            for _ in self.match_condition_recursive(tree, nodes_option, backrefs_map, 0):
                yield True

    def could(self, tree, nodes, backrefs_map, position, parent):
        if not self.allow_more:
            raise NotImplementedError()

        if len(nodes) < len(self.conditions):
            to_add = len(self.conditions) - len(nodes)
            tree.append(['*'] * to_add, [''] * to_add, ['_'] * to_add, ['_'] * to_add, [[] for _ in range(to_add)], [parent] * to_add, ['_'] * to_add)
            new_nodes = tree.children(parent)
            nodes_to_remove = set(new_nodes).difference(nodes)
            if not tree.is_valid(position):
                tree.delete(nodes_to_remove)
                return
            tree.delete(nodes_to_remove)
        for nodes_option in self._get_nodes_options(nodes, support_new=True):
            for cost in self.could_condition_recursive(tree, nodes_option, backrefs_map, position, 0):
                yield cost + nodes_option.count(-1)
        # Case of good asignment of descandants doesn't exist


class HasSiblingsList(TreePattern):
    def __init__(self, multi_id):
        self.condition = HasSiblingsListInner(multi_id)

    def match(self, tree, node, backrefs_map):
        siblings = tree.right_siblings(node)
        return self.condition.match(tree, siblings, backrefs_map)

    def could(self, tree, node, backrefs_map, position):
        siblings = tree.right_siblings(node)
        return self.condition.could(tree, siblings, backrefs_map, position, brother=node)


class HasSiblingsListInner(TreePattern):
    def __init__(self, multi_id):
        self.conditions = multi_id.conditions
        self.allow_more = multi_id.allow_more
        assert self.allow_more is False, 'Allow more is not supported yet'

    def match_condition_recursive(self, tree, nodes, backrefs_map, i):
        if i == len(self.conditions):
            yield True
            return

        old_map = backrefs_map.copy()
        for _ in self.conditions[i].match(tree, nodes[i], backrefs_map):
            for _ in self.match_condition_recursive(tree, nodes, backrefs_map, i + 1):
                yield True
            backrefs_map.clear()
            backrefs_map.update(old_map)

    def could_condition_recursive(self, tree, nodes, backrefs_map, position, i):
        if i == len(self.conditions):
            yield 0
            return

        old_map = backrefs_map.copy()
        for my_cost in self.conditions[i].could(tree, nodes[i], backrefs_map, position):
            for other_cost in self.could_condition_recursive(tree, nodes, backrefs_map, position, i + 1):
                yield my_cost + other_cost
            backrefs_map.clear()
            backrefs_map.update(old_map)

    def match(self, tree, nodes, backrefs_map):
        if len(nodes) != len(self.conditions) and not self.allow_more:
            return
        for _ in self.match_condition_recursive(tree, nodes, backrefs_map, 0):
            yield True
        return True

    def could(self, tree, nodes, backrefs_map, position, brother):
        edit_cost = 0
        nodes_to_remove = set()
        parent = tree.heads(brother)
        if parent == 0:
            return
        elif len(nodes) < len(self.conditions) and parent != -1:
            to_add = len(self.conditions) - len(nodes)
            tree.append(['*'] * to_add, [''] * to_add, ['_'] * to_add, ['_'] * to_add, [[] for _ in range(to_add)], [parent] * to_add, ['_'] * to_add)
            edit_cost += to_add
            old_siblings = set(nodes)
            nodes = tree.right_siblings(brother)
            nodes_to_remove = set(nodes).difference(old_siblings)
            if not tree.is_valid(position):
                tree.delete(nodes_to_remove)
                return
        elif parent == -1:
            nodes = [-1] * len(self.conditions)
            edit_cost = len(nodes)
        elif len(nodes) > len(self.conditions):
            return
        for cost in self.could_condition_recursive(tree, nodes, backrefs_map, position, 0):
            yield edit_cost + cost
        if nodes_to_remove:
            tree.delete(nodes_to_remove)

class HasAdjacentChild(TreePattern):
    def __init__(self, condition):
        self.condition = condition

    def match(self, tree, node, backrefs_map):
        for child in tree.children(node):
            if (child - node) not in [-1, +1]:
                continue
            if self.condition.match(tree, child, backrefs_map):
                return True
        return False

## ----------------------------------------------------------------------------
#                                   Parents

class HasLeftHead(TreePattern):
    def __init__(self, condition):
        self.condition = condition

    def match(self, tree, node, backrefs_map):
        if node == 0:
            return False

        head = tree.heads(node)
        return head < node and self.condition.match(tree, head, backrefs_map)

class HasRightHead(TreePattern):
    def __init__(self, condition):
        self.condition = condition

    def match(self, tree, node, backrefs_map):
        if node == 0:
            return False

        head = tree.heads(node)
        return head > node and self.condition.match(tree, head, backrefs_map)

class HasHead(TreePattern):
    def __init__(self, condition):
        self.condition = condition

    def match(self, tree, node, backrefs_map):
        if node == 0:
            return False

        head = tree.heads(node)
        return self.condition.match(tree, head, backrefs_map)

class HasPredecessor(TreePattern):
    def __init__(self, condition):
        self.condition = condition

    def match(self, tree, node, backrefs_map):
        while True:
            node = tree.heads(node)
            if self.condition.match(tree, node, backrefs_map):
                return True
            if node == 0:
                break
        return False

class HasAdjacentLeftHead(TreePattern):
    def __init__(self, condition):
        self.condition = condition

    def match(self, tree, node, backrefs_map):
        if node == 0:
            return False

        head = tree.heads(node)
        adjacent = (head + 1 == node)
        return adjacent and self.condition.match(tree, head, backrefs_map)

class HasAdjacentRightHead(TreePattern):
    def __init__(self, condition):
        self.condition = condition

    def match(self, tree, node, backrefs_map):
        if node == 0:
            return False

        head = tree.heads(node)
        adjacent = (head - 1 == node)
        return adjacent and self.condition.match(tree, head, backrefs_map)

class HasAdjacentHead(TreePattern):
    def __init__(self, condition):
        self.condition = condition

    def match(self, tree, node, backrefs_map):
        if node == 0:
            return False

        head = tree.heads(node)
        adjacent = (head - node) in [-1, +1]
        return adjacent and self.condition.match(tree, head, backrefs_map)

## ----------------------------------------------------------------------------
#                                 Neighbors

class HasLeftNeighbor(TreePattern):
    def __init__(self, condition):
        self.condition = condition

    def match(self, tree, node, backrefs_map):
        if node == 0:
            return False

        for neighbor in range(0, node):
            if self.condition.match(tree, neighbor, backrefs_map):
                return True
        return False

class HasRightNeighbor(TreePattern):
    def __init__(self, condition):
        self.condition = condition

    def match(self, tree, node, backrefs_map):
        for neighbor in range(node + 1, len(tree) + 1):
            if self.condition.match(tree, neighbor, backrefs_map):
                return True
        return False

class HasAdjacentLeftNeighbor(TreePattern):
    def __init__(self, condition):
        self.condition = condition

    def match(self, tree, node, backrefs_map):
        if node == 0:
            return False

        neighbor = node - 1
        return self.condition.match(tree, neighbor, backrefs_map)

class HasAdjacentRightNeighbor(TreePattern):
    def __init__(self, condition):
        self.condition = condition

    def match(self, tree, node, backrefs_map):
        if node == len(tree):
            return False

        neighbor = node + 1
        return self.condition.match(tree, neighbor, backrefs_map)

## ----------------------------------------------------------------------------
#                            Misc. tree structure

class CanHead(TreePattern):
    def __init__(self, backref):
        self.backref = backref

    def match(self, tree, node, backrefs_map):
        if self.backref not in backrefs_map:
            return False

        head = node
        child = backrefs_map[self.backref]
        return head not in [child] + tree.children_recursive(child)

class CanBeHeadedBy(TreePattern):
    def __init__(self, backref):
        self.backref = backref

    def match(self, tree, node, backrefs_map):
        if self.backref not in backrefs_map:
            return False

        head = backrefs_map[self.backref]
        child = node
        return head not in [child] + tree.children_recursive(child)

class IsRoot(TreePattern):
    def match(self, tree, node, backrefs_map):
        return node == 0

class NotRoot(TreePattern):
    def __init__(self, condition):
        self.condition = condition

    def match(self, tree, node, backrefs_map):
        if node == 0:
            return
        for _ in self.condition.match(tree, node, backrefs_map):
            yield True

    def could(self, tree, node, backrefs_map, position):
        if node == 0:
            return
        for cost in self.condition.could(tree, node, backrefs_map, position):
            yield cost

class IsTop(TreePattern):
    def match(self, tree, node, backrefs_map):
        return node != 0 and tree.heads(node) == 0

class IsLeaf(TreePattern):
    def match(self, tree, node, backrefs_map):
        return not tree.children(node)

## ----------------------------------------------------------------------------
#                                 Attributes

class AttrMatches(TreePattern):
    def __init__(self, attr, pred_fn):
        self.attr = attr
        self.pred_fn = pred_fn

    def match(self, tree, node, backrefs_map):
        if node == 0:
            return False

        attr = getattr(tree, self.attr)(node)
        if self.pred_fn(attr, backrefs_map=backrefs_map) == 0:
            yield True

    def could(self, tree, node, backrefs_map, position):
        if node == 0:
            return

        if node == -1:
            yield 0
            return

        attr = getattr(tree, self.attr)(node)
        yield tree.SOFT_UNIT * self.pred_fn(attr, backrefs_map=backrefs_map)

class FeatsMatch(TreePattern):
    def __init__(self, pred_fn):
        self.pred_fn = pred_fn

    def match(self, tree, node, backrefs_map):
        if node == 0:
            return False

        attr = '|'.join(tree.feats(node))
        return self.pred_fn(attr)

## ----------------------------------------------------------------------------
#                                   Logic

class And(TreePattern):
    def __init__(self, conditions):
        self.conditions = conditions

    def match_condition_recursive(self, tree, node, backrefs_map, i):
        if i == len(self.conditions):
            yield True
            return

        old_map = backrefs_map.copy()
        for _ in self.conditions[i].match(tree, node, backrefs_map):
            for _ in self.match_condition_recursive(tree, node, backrefs_map, i + 1):
                yield True
            backrefs_map.clear()
            backrefs_map.update(old_map)

    def could_condition_recursive(self, tree, node, backrefs_map, position, i):
        if i == len(self.conditions):
            yield 0
            return

        old_map = backrefs_map.copy()
        for my_cost in self.conditions[i].could(tree, node, backrefs_map, position):
            for other_cost in self.could_condition_recursive(tree, node, backrefs_map, position, i + 1):
                yield my_cost + other_cost
            backrefs_map.clear()
            backrefs_map.update(old_map)

    def match(self, tree, node, backrefs_map):
        for _ in self.match_condition_recursive(tree, node, backrefs_map, 0):
            yield True

    def could(self, tree, node, backrefs_map, position):
        for total_cost in self.could_condition_recursive(tree, node, backrefs_map, position, 0):
            yield total_cost

class Or(TreePattern):
    def __init__(self, conditions):
        self.conditions = conditions

    def match(self, tree, node, backrefs_map):
        for condition in self.conditions:
            if condition.match(tree, node, backrefs_map):
                return True
        return False

class Not(TreePattern):
    def __init__(self, condition):
        self.condition = condition

    def match(self, tree, node, backrefs_map):
        # If sub-condition matchesm 'not sub-condition' doesn't. Sub-condition
        # might modify the backrefs_map on successful match, but since
        # 'not sub-condition' doesn't match, these changes shouldn't be visible
        # to the outside world.
        copy = backrefs_map.copy()
        return not self.condition.match(tree, node, copy)

class AlwaysTrue(TreePattern):
    def match(self, tree, node, backrefs_map):
        yield True

    def could(self, tree, node, backrefs_map, position):
        yield 0

## ----------------------------------------------------------------------------
#                                  Backrefs

class SetBackref(TreePattern):
    def __init__(self, backref, condition):
        self.backref = backref
        self.condition = condition

    def match(self, tree, node, backrefs_map):
        backref_node = backrefs_map.get(self.backref)

        if backref_node is None:
            # Update the backref so the underlying condition can see it.
            backrefs_map[self.backref] = node
        else:
            node = backref_node

        # If condition fails, undo the changes to backrefs_map.
        for _ in self.condition.match(tree, node, backrefs_map):
            yield True

        # if backref_node is None:
        #     # If I added the key, delete it.
        #     del backrefs_map[self.backref]

    def could(self, tree, node, backrefs_map, position):
        backref_node = backrefs_map.get(self.backref)

        if backref_node is None:
            # Update the backref so the underlying condition can see it.
            backrefs_map[self.backref] = node
        else:
            node = backref_node

        # If condition fails, undo the changes to backrefs_map.
        for cost in self.condition.could(tree, node, backrefs_map, position):
            yield cost

class EqualsBackref(TreePattern):
    def __init__(self, backref):
        self.backref = backref

    def match(self, tree, node, backrefs_map):
        return backrefs_map.get(self.backref) == node
