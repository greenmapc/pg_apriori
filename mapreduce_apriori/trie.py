class TrieNode(object):
    def __init__(self, item, depth, items):
        self.item = item
        self.depth = depth
        self.items = items
        self.support = 0
        self.children = []
        self.invalid = False
        self.word_finished = False

class Child(object):
    def __init__(self, node):
        self.node = node
        self.next = None
        self.prev = None

    def add_next(self, next):
        next.prev = self.node
        self.next = next

class TrieChildren(object):

    def __init__(self):
        self.last = None
        self.list = []

    def add_child(self, child):
        if self.last is None:
            self.last = child
        else:
            self.last.next = child
            child.prev = self.last.next
            self.last = child


def binary_search(array, target):
    lower = 0
    upper = len(array)
    if upper == lower:
        return None
    while lower < upper:
        x = lower + (upper - lower) // 2
        val = array[x].item
        if target == val:
            return array[x]
        elif target > val:
            if lower == x:
                break
            lower = x
        elif target < val:
            upper = x
    for i in (lower, upper - 1):
        if array[i].item == target:
            return array[i]
    return None


def find_frequent_itemsets(node, support):
    if node.word_finished:
        if node.support < support:
            node.invalid = True
            return []
        else:
            return [(node.items, node.support)]
    else:
        result = []
        for i in range(len(node.children) - 1, -1, -1):
            nodes = find_frequent_itemsets(node.children[i], support)
            result.extend(nodes)
            if node.children[i].invalid:
                node.children.remove(node.children[i])
        if len(node.children) == 0:
            node.invalid = True
        return result

def count_support(node, target, iterator):
    if node.items == target:
        node.support += 1
        return
    current_item = next(iterator)
    node = binary_search(node.children, current_item)
    if node:
        count_support(node, target, iterator)


def add(root, items):
    current_node = root
    for item in items:
        found_node = binary_search(current_node.children, item)
        if found_node is not None:
            current_node = found_node
        else:
            new_node = TrieNode(item, current_node.depth + 1, current_node.items + [item])
            current_node.children.append(new_node)
            current_node = new_node
    # last
    current_node.word_finished = True


def search_candidates(visited, node, max_depth, used_candidates_items, candidate_items):
    node.support = 0
    if node.word_finished:
        node.word_finished = False
        if candidate_items:
            for item in candidate_items:
                if node.item != item:
                    add(node, [item])
        else:
            node.invalid = True
        return node.item
    if node.depth == max_depth - 1 or max_depth == 1:
        for i in range(len(node.children) - 1, -1, -1):
            neighbor = node.children[i]
            candidate = search_candidates(visited, neighbor, max_depth, used_candidates_items, candidate_items)
            if not candidate in used_candidates_items:
                used_candidates_items.add(candidate)
                candidate_items.insert(0, candidate)
            if neighbor.invalid:
                node.children.remove(neighbor)
    else:
        if node not in visited:
            visited.add(node)
            for i in range(len(node.children)):
                neighbor = node.children[i]
                search_candidates(visited, neighbor, max_depth, set(), list())
                if neighbor.invalid:
                    node.children.remove(neighbor)
    if len(node.children) == 0:
        node.invalid = True


def dfs(visited, node):
    if node not in visited:
        if node.word_finished:
            print(node.items, node.support)
        visited.add(node)
        for neighbor in node.children:
            dfs(visited, neighbor)

# frequent_one = [(['BLACK'], 1), (['MBE'], 2), (['NON_MINORITY'], 4), (['WBE'], 3), (['ZZZ'], 5)]
# current_candidates_tree = TrieNode(None, 0, [])
# for candidate in frequent_one:
#     add(current_candidates_tree, candidate[0])
# k = 2
# example = ['NON_MINORITY', 'WBE', 'ZZZ']
# while current_candidates_tree and k <= 3:
#     search_candidates(set(), current_candidates_tree, k - 1, set(), list())
#     # todo add count support
#     dfs(set(), current_candidates_tree)
#     k += 1
#     print("end iteration")
# iterator = iter(example)
# count_support(current_candidates_tree, example, iterator)
# dfs(set(), current_candidates_tree)
