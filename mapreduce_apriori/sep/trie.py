class TrieNode(object):
    def __init__(self, item, depth, items):
        self.item = item
        self.depth = depth
        self.items = items
        self.support = 0
        self.children = []
        self.invalid = False
        self.word_finished = False


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
            nodes_for_remove = []
            for i in range(len(node.children)):
                neighbor = node.children[i]
                search_candidates(visited, neighbor, max_depth, set(), list())
                if neighbor.invalid:
                    nodes_for_remove.append(neighbor)
            node.children = [x for x in node.children if (x not in nodes_for_remove)]
    if len(node.children) == 0:
        node.invalid = True


def dfs(visited, node):
    if node not in visited:
        if node.word_finished:
            print(node.items, node.support)
        visited.add(node)
        for neighbor in node.children:
            dfs(visited, neighbor)
