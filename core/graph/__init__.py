"""
Модуль управления графом мыслей
Экспортирует главный экземпляр менеджера графа
"""

from .manager import GraphManager
from config import GRAPH_FILE_PATH

# Создаём глобальный экземпляр менеджера графа
graph_manager = GraphManager(graph_path=GRAPH_FILE_PATH)

__all__ = ['graph_manager', 'GraphManager']
