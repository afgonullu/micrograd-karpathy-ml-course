from graphviz import Digraph

def trace(root):
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges

def draw_dot(root, format='svg', rankdir='LR', show_label=True):
    """
    format: png | svg | ...
    rankdir: TB (top to bottom graph) | LR (left to right)
    """
    assert rankdir in ['LR', 'TB']
    nodes, edges = trace(root)
    dot = Digraph(format=format, graph_attr={'rankdir': "LR"}) #, node_attr={'rankdir': 'TB'})

    for n in nodes:
        uid = str(id(n))
        dot.node(name=uid, label = "{ %s | data %.4f | grad %.4f }" % (n.label,n.data,n.grad) if show_label else "{ data %.4f | grad %.4f }" % (n.data,n.grad), shape='record')
        if n._op:
            dot.node(name=uid + n._op, label=n._op)
            dot.edge(uid + n._op, uid)

    for n1, n2 in edges:
        uid1 = str(id(n1))
        uid2 = str(id(n2))
        dot.edge(uid1, uid2 + n2._op)

    return dot

def draw_nn(model, format='svg', rankdir='LR', input_labels=None, layer_labels=None, show_weights="both", show_bias=False, precision=2, weight_position='neuron', weight_threshold=None, color_edges=False, inputs=None, show_node_values=True):
    """
    Render a high-level neural network graph (layers and neurons) for an nn.MLP-like model.

    Parameters
    ----------
    model : nn.MLP or compatible
        The network to visualize. Must expose `layers`, where each layer has `neurons`,
        and each neuron has list `w` (weights) and scalar `b` (bias).
    format : str
        Output format for Graphviz (e.g., 'svg', 'png').
    rankdir : str
        'LR' for left-to-right or 'TB' for top-to-bottom layout.
    input_labels : list[str] | None
        Optional labels for input features. If None and `inputs` provided, uses their numeric values; else X1..Xm.
    inputs : list[engine.Value] | list[float] | None
        Optional inputs used for the most recent forward pass. When provided (or when model caches them), input nodes will show values.
    show_node_values : bool
        If True, neuron nodes display { label | data | grad } similar to draw_dot. Requires a recent forward/backward pass to populate grads.
    layer_labels : list[str] | None
        Optional custom titles for each layer cluster (same length as model.layers).
    show_weights : bool | str
        If True or 'value', label edges with numeric weight values.
        If 'label', use parameter labels (e.g., w0_0). If 'both', use label=value.
    weight_position : str
        Where to place weight labels: 'center' (default), 'head', 'tail', 'neuron', or 'none'.
    weight_threshold : float | None
        Only show weight labels (and color emphasis) when abs(weight) >= threshold. If None, show all.
    color_edges : bool
        If True, color edges by sign (blue for +, red for -) and scale penwidth by magnitude.
    show_bias : bool
        If True, add a small bias node feeding each neuron (optionally labeled with value).
    precision : int
        Number of decimal places for numeric labels.
    """
    assert rankdir in ['LR', 'TB']

    # infer sizes
    assert len(model.layers) > 0, "Model has no layers to draw"
    num_inputs = len(model.layers[0].neurons[0].w)
    layer_sizes = [len(layer.neurons) for layer in model.layers]

    # Defaults for labels
    # Try to get inputs from model cache if not provided
    cached_inputs = getattr(model, 'last_inputs', None)
    if inputs is None and cached_inputs is not None:
        inputs = cached_inputs

    # Build input labels
    if input_labels is None:
        if show_node_values and inputs is not None:
            try:
                vals = [xi.data if hasattr(xi, 'data') else float(xi) for xi in inputs]
                input_labels = [f"{v:.{precision}f}" for v in vals]
            except Exception:
                input_labels = [f"X{i+1}" for i in range(num_inputs)]
        else:
            input_labels = [f"X{i+1}" for i in range(num_inputs)]
    if layer_labels is None:
        layer_labels = [f"Layer {i+1}" for i in range(len(model.layers))]

    dot = Digraph(format=format, graph_attr={
        'rankdir': rankdir,
        'newrank': 'true',
        'splines': 'spline',
        'nodesep': '1.0',
        'ranksep': '1.6'
    })

    # Common styling
    dot.attr('node', shape='circle', fontsize='10', fixedsize='true', width='0.6')
    dot.attr('edge', arrowsize='0.6', labelfontsize='9')

    # 1) Input layer
    input_node_ids = []
    with dot.subgraph(name='cluster_input') as c:
        c.attr(label='Input Layer')
        c.attr(rank='same')
        base_xlabels = {}
        for idx, lbl in enumerate(input_labels):
            node_id = f"x{idx}"
            input_node_ids.append(node_id)
            if show_node_values and inputs is not None:
                val = inputs[idx]
                data = val.data if hasattr(val, 'data') else float(val)
                grad = getattr(val, 'grad', 0.0)
                c.node(node_id, label=f"{data:.{precision}f}", xlabel=f"grad {grad:.{precision}f}")
                base_xlabels[node_id] = f"grad {grad:.{precision}f}"
            else:
                c.node(node_id, label=lbl)

    # 2) Hidden/Output layers
    prev_ids = input_node_ids
    layer_anchor_ids = []
    # For neuron-level labels, collect info while drawing edges
    neuron_xlabels = {}
    for li, layer in enumerate(model.layers):
        is_last = li == len(model.layers) - 1
        layer_name = layer_labels[li] if li < len(layer_labels) else f"Layer {li+1}"
        cluster_name = f"cluster_layer_{li}"

        neuron_ids = []
        with dot.subgraph(name=cluster_name) as c:
            c.attr(label=f"{layer_name}{' (Output)' if is_last else ''}")
            c.attr(rank='same')

            for ni, neuron in enumerate(layer.neurons):
                node_id = f"h{li}_{ni}" if not is_last else f"o{ni}"
                neuron_ids.append(node_id)

                label = f"Y{ni+1}" if not is_last else f"Z{ni+1}"
                if show_node_values:
                    sum_val = getattr(neuron, 'last_sum', None)
                    out_val = getattr(neuron, 'last_out', None)
                    pre_data = sum_val.data if sum_val is not None else 0.0
                    post_data = out_val.data if out_val is not None else 0.0
                    post_grad = getattr(out_val, 'grad', 0.0) if out_val is not None else 0.0
                    xlabel_text = f"pre {pre_data:.{precision}f} | post {post_data:.{precision}f}\ngrad {post_grad:.{precision}f}"
                    c.node(node_id, label=label, xlabel=xlabel_text)
                    base_xlabels[node_id] = xlabel_text
                else:
                    c.node(node_id, label=label)

                # Optional bias node feeding into neuron
                if show_bias:
                    b_node_id = f"b{li}_{ni}"
                    if show_weights in ['both', True, 'value']:
                        b_label = f"b={neuron.b.data:.{precision}f}"
                    elif show_weights == 'label':
                        b_label = 'b'
                    else:
                        b_label = 'b'
                    c.node(b_node_id, label=b_label, shape='point', width='0.1', height='0.1')
                    c.edge(b_node_id, node_id, style='dashed', constraint='false')

            # Add invisible anchor per layer to help keep clusters side-by-side
            anchor_id = f"layer_anchor_{li}"
            layer_anchor_ids.append(anchor_id)
            c.node(anchor_id, label='', style='invis', width='0.01', height='0.01')

        # Connect from previous layer to this layer
        for ni, neuron in enumerate(layer.neurons):
            for pi, prev_node_id in enumerate(prev_ids):
                w = neuron.w[pi]
                if show_weights in [True, 'value']:
                    edge_label = f"{w.data:.{precision}f}"
                elif show_weights == 'label':
                    edge_label = w.label if isinstance(w.label, str) else None
                elif show_weights == 'both':
                    edge_label = f"{w.label}={w.data:.{precision}f}"
                else:
                    edge_label = None
                # threshold gating
                show_label = edge_label is not None
                if weight_threshold is not None and edge_label is not None:
                    show_label = abs(w.data) >= weight_threshold

                # edge styling
                edge_kwargs = {}
                if color_edges:
                    color = 'steelblue' if w.data >= 0 else 'indianred'
                    width = 0.8 + 2.0 * (abs(w.data))
                    edge_kwargs.update({'color': color, 'penwidth': str(width)})

                if show_label and show_weights and weight_position != 'none' and weight_position != 'neuron':
                    if weight_position == 'head':
                        dot.edge(prev_node_id, neuron_ids[ni], headlabel=edge_label, labeldistance='1.6', labelangle='25', decorate='true', **edge_kwargs)
                    elif weight_position == 'tail':
                        dot.edge(prev_node_id, neuron_ids[ni], taillabel=edge_label, labeldistance='1.6', labelangle='-25', decorate='true', **edge_kwargs)
                    else: # center
                        dot.edge(prev_node_id, neuron_ids[ni], label=edge_label, decorate='true', **edge_kwargs)
                else:
                    dot.edge(prev_node_id, neuron_ids[ni], **edge_kwargs)

                if show_label and show_weights and weight_position == 'neuron':
                    nid = neuron_ids[ni]
                    if nid not in neuron_xlabels:
                        neuron_xlabels[nid] = []
                    neuron_xlabels[nid].append(edge_label)

        prev_ids = neuron_ids

    # If neuron-level labels requested, attach them as xlabels on neuron nodes
    if show_weights and weight_position == 'neuron' and neuron_xlabels:
        for nid, labels in neuron_xlabels.items():
            text = "\n".join(labels)
            if nid in base_xlabels:
                text = f"{text}\n{base_xlabels[nid]}"
            dot.node(nid, xlabel=text)

    # Connect anchors invisibly to force horizontal ordering of clusters
    for i in range(len(layer_anchor_ids) - 1):
        dot.edge(layer_anchor_ids[i], layer_anchor_ids[i+1], style='invis', weight='100')

    return dot