"""Tests for Node class."""

import numpy as np
import pytest
from planning.graph.node import Node, distance, steer, get_nearest_node


def test_node_creation_2d():
    """Test creating a 2D node."""
    node = Node(state=(1.0, 2.0), cost=5.0)
    
    assert node.dim == 2
    assert node[0] == 1.0
    assert node[1] == 2.0
    assert node.cost == 5.0
    assert node.parent is None


def test_node_creation_3d():
    """Test creating a 3D node."""
    node = Node(state=(1.0, 2.0, 3.0))
    
    assert node.dim == 3
    assert node[0] == 1.0
    assert node[2] == 3.0


def test_parent_child_relationship():
    """Test parent-child node relationships."""
    parent = Node(state=(0, 0))
    child = Node(state=(1, 1), parent=parent)
    
    assert child.parent == parent
    assert child in parent.children
    assert parent.is_root()
    assert not child.is_root()


def test_distance_calculation():
    """Test distance between nodes."""
    node1 = Node(state=(0, 0))
    node2 = Node(state=(3, 4))
    
    dist = distance(node1, node2)
    
    assert dist == 5.0  # 3-4-5 triangle


def test_path_extraction():
    """Test path extraction from node to root."""
    root = Node(state=(0, 0), cost=0)
    node1 = Node(state=(1, 0), parent=root, cost=1)
    node2 = Node(state=(2, 0), parent=node1, cost=2)
    
    path_from_root = node2.get_path_from_root()
    
    assert len(path_from_root) == 3
    assert path_from_root[0] == root
    assert path_from_root[-1] == node2


def test_steer_within_distance():
    """Test steering when target is within max distance."""
    from_node = Node(state=(0, 0))
    to_node = Node(state=(1, 0))
    
    new_node = steer(from_node, to_node, max_distance=2.0)
    
    assert np.allclose(new_node.state, [1, 0])


def test_steer_exceeds_distance():
    """Test steering when target exceeds max distance."""
    from_node = Node(state=(0, 0))
    to_node = Node(state=(10, 0))
    
    new_node = steer(from_node, to_node, max_distance=2.0)
    
    assert np.allclose(new_node.state, [2, 0])
    assert new_node.parent == from_node


def test_get_nearest_node():
    """Test finding nearest node."""
    nodes = [
        Node(state=(0, 0)),
        Node(state=(1, 0)),
        Node(state=(0, 1)),
    ]
    target = Node(state=(0.9, 0.1))
    
    nearest = get_nearest_node(nodes, target)
    
    assert np.allclose(nearest.state, [1, 0])


def test_cost_update():
    """Test cost update propagation."""
    root = Node(state=(0, 0), cost=0)
    child1 = Node(state=(1, 0), parent=root, cost=1)
    child2 = Node(state=(2, 0), parent=child1, cost=2)
    
    child1.update_cost(0.5)
    
    assert child1.cost == 0.5
    assert child2.cost == 1.5  # Updated by difference


def test_node_depth():
    """Test node depth calculation."""
    root = Node(state=(0, 0))
    child = Node(state=(1, 0), parent=root)
    grandchild = Node(state=(2, 0), parent=child)
    
    assert root.get_depth() == 0
    assert child.get_depth() == 1
    assert grandchild.get_depth() == 2

