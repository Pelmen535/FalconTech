"""Bound TensorRT execution contexts retained by CUDA graphs.

Keep the online batch-one graph and one bulk graph. Eviction synchronizes the
stream and destroys graph executable/graph before releasing TensorRT contexts.
No input, output, precision, engine, or numerical operation changes.
"""

def make_room(graphs, incoming, synchronize, destroy_exec, destroy_graph, capacity=2):
    if capacity < 1:
        raise ValueError('Graph cache capacity must be positive')
    if incoming in graphs:
        return None
    if len(graphs) < capacity:
        return None
    # Dict order is creation order. Prefer evicting a bulk shape over online B1.
    victim=next((key for key in graphs if key != 1), next(iter(graphs)))
    contexts,graph,executable=graphs[victim]
    synchronize()
    # Keep ownership if CUDA destruction fails; the caller must fail closed.
    destroy_exec(executable)
    destroy_graph(graph)
    del graphs[victim]
    contexts.clear()
    return victim
