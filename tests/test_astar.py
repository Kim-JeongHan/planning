import numpy as np

from planning.graph import Graph, Node
from planning.search import AStar


def make_simple_graph():
    """사각형 형태의 테스트 그래프 생성"""
    n1 = Node((0, 0))
    n2 = Node((1, 0))
    n3 = Node((1, 1))
    n4 = Node((0, 1))

    graph = Graph()
    for n in [n1, n2, n3, n4]:
        graph.add_node(n)

    # Edge 비용 설정
    graph.add_edge(n1, n2, 1.0)
    graph.add_edge(n2, n3, 2.0)
    graph.add_edge(n1, n4, 1.5)
    graph.add_edge(n4, n3, 0.5)

    return graph, n1, n2, n3, n4


def test_astar_shortest_path():
    """A*가 올바른 최단 경로를 찾는지 테스트"""
    graph, n1, _, n3, n4 = make_simple_graph()
    planner = AStar(graph)
    path = planner.search(n1, n3)

    # 예상 경로는 n1 → n4 → n3
    expected = [n1, n4, n3]
    assert path == expected

    # 총 비용 계산
    total_cost = 0.0
    for i in range(len(path) - 1):
        for edge in graph.edges:
            if (edge.node1 == path[i] and edge.node2 == path[i + 1]) or (
                edge.node2 == path[i] and edge.node1 == path[i + 1]
            ):
                total_cost += edge.cost
    assert np.isclose(total_cost, 2.0)


def test_astar_no_path():
    """연결되지 않은 노드에 대해 빈 경로 반환"""
    graph, n1, _, _, _ = make_simple_graph()
    isolated = Node(state=(5, 5))
    graph.add_node(isolated)
    planner = AStar(graph)

    path = planner.search(n1, isolated)
    assert path == []


def test_astar_start_is_goal():
    """시작 노드와 목표 노드가 같을 때"""
    graph, n1, _, _, _ = make_simple_graph()
    planner = AStar(graph)

    path = planner.search(n1, n1)
    assert path == [n1]
