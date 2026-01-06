"""Скрипт для анализа ролей узлов в графе"""
import pickle
from collections import Counter

GRAPH_FILE = "logs/mind_graph.gpickle"

with open(GRAPH_FILE, 'rb') as f:
    graph = pickle.load(f)

# Собираем все роли
roles = [attrs.get('role', 'NO_ROLE_ATTR') for _, attrs in graph.nodes(data=True)]
role_counts = Counter(roles)

print(f"Всего узлов: {graph.number_of_nodes()}")
print(f"\nУникальные роли ({len(role_counts)}):")
for role, count in role_counts.most_common():
    print(f"  {role!r}: {count}")
