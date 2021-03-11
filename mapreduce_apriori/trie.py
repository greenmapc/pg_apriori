class TrieNode(object):
    def __init__(self, item, depth, items):
        self.item = item
        self.depth = depth
        self.items = items
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
            print(node.items)
        visited.add(node)
        for neighbor in node.children:
            dfs(visited, neighbor)


# check
if __name__ == "__main__":
    root = TrieNode('*', 0, [])
    add(root, "acd")
    add(root, 'acg')
    add(root, 'ach')
    add(root, 'aeg')
    add(root, 'aem')

    dfs(set(), root)
    search_candidates(set(), root, 2, set(), list())
    print('000')
    dfs(set(), root)

