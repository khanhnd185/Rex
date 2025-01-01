
class Graph:
  def __init__(self):
    self.graph = {}

  def add_vertex(self, vertex_name, parent_vertexes=None, data=None):
    if vertex_name not in self.graph:
      self.graph[vertex_name] = {'data': data, 'parents': []}
      if parent_vertexes:
        for parent_vertex in parent_vertexes:
          self.add_edge(parent_vertex, vertex_name)
      else:
        self.root_name = vertex_name

  def add_edge(self, from_vertex, to_vertex):
    if from_vertex in self.graph and to_vertex in self.graph:
      self.graph[to_vertex]['parents'].append(from_vertex)

  def get_vertices(self):
    return list(self.graph.keys())

  def get_edges(self):
    edges = []
    for vertex in self.graph:
      for neighbor in self.graph[vertex]['parents']:
        edges.append((vertex, neighbor))
    return edges

  def get_parents(self, vertex_name):
    if vertex_name in self.graph:
      return self.graph[vertex_name]['parents']
    else:
      return []

  def get_data(self, vertex_name):
    if vertex_name in self.graph:
      return self.graph[vertex_name]['data']
    else:
      return None

  def summary(self):
    result = "Vertices:\n"
    result += ',\n'.join(vertex + ': ' + str(self.graph[vertex]['data']) for vertex in self.graph) + "\n"
    result += "Edges:\n"
    result += ',\n'.join(f"({vertex}, {neighbor})" for vertex, neighbor in self.get_edges())
    return result


if __name__ == "__main__":
  graph = Graph()

  graph.add_vertex('A', data=10)
  graph.add_vertex('B', ['A'], data=20)
  graph.add_vertex('C', ['A'], data=30)
  graph.add_vertex('D', ['B','A'], data=40)

  print("Vertices:", graph.get_vertices())
  print("Edges:", graph.get_edges())
  print("Parent of 'D':", graph.get_parents('D'))
  print("Parent of 'C':", graph.get_parents('C'))
  print("Data of 'B':", graph.get_data('B'))

  print("\nGraph representation:")
  print(graph.summary())
