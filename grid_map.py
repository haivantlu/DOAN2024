import heapq

class GM:
    def __init__(self, grid):
        self.grid = grid

    def neighbors(self, current):
        x, y = current
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]  # Right, down, left, up, and the diagonal directions
        neighbors = []

        for dx, dy in directions:
            new_x, new_y = x + dx, y + dy
            if 0 <= new_x < len(self.grid) and 0 <= new_y < len(self.grid[0]):
                if self.grid[new_x][new_y] != 1:  # Check the target cell
                    if dx != 0 and dy != 0:  # If moving diagonally
                        if self.grid[x + dx][y] != 1 and self.grid[x][y + dy] != 1:  # Check the two adjacent cells
                            neighbors.append((new_x, new_y))
                    else:  # If moving horizontally or vertically
                        neighbors.append((new_x, new_y))

        return neighbors

    def cost(self, current, next):
        x1, y1 = current
        x2, y2 = next
        return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5  # Euclidean distance

    def heuristic(self, goal, next):
        x1, y1 = goal
        x2, y2 = next
        return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5  # Euclidean distance

    def create_individual(self, start, goal):
        frontier = []
        heapq.heappush(frontier, (0, start))
        came_from = {}
        cost_so_far = {}
        came_from[start] = None
        cost_so_far[start] = 0

        while frontier:
            current = heapq.heappop(frontier)[1]

            if current == goal:
                break

            for next in self.neighbors(current):
                new_cost = cost_so_far[current] + self.cost(current, next)
                if next not in cost_so_far or new_cost < cost_so_far[next]:
                    cost_so_far[next] = new_cost
                    priority = new_cost + self.heuristic(goal, next)
                    heapq.heappush(frontier, (priority, next))
                    came_from[next] = current

        if goal not in came_from:
            return []  # Return an empty list if there is no path to the goal

        # Reconstruct the path
        current = goal
        path = []
        while current is not None:
            path.append(current)
            current = came_from[current]
        path.reverse()

        return path