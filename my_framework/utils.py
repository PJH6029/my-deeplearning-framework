import os, subprocess, tempfile
from typing import Any, Union
from numpy.typing import NDArray

import my_framework.core.base as base

# =============================================================================
# visualization tools
# =============================================================================
def _dot_var(v: base.Variable, verbose=False) -> str:
    name = "" if v.name is None else v.name
    if verbose and v.data is not None:
        if v.name is not None:
            name += ": "
        name += f"{str(v.shape)} {str(v.dtype)}"
    return f'{id(v)} [label="{name}", color=orange, style=filled]\n'

def _dot_func(f: base.Function) -> str:
    txt = f"{id(f)} [label={f.__class__.__name__}, color=lightblue, style=filled, shape=box]\n"
    
    dot_edge = lambda v, w: f"{id(v)} -> {id(w)}\n"
    for x in f.inputs:
        txt += dot_edge(x, f)
    for y in f.outputs:
        txt += dot_edge(f, y()) # y is weakref
    return txt

def get_dot_graph(output: base.Variable, verbose=True) -> str:
    txt = ""
    funcs: list[base.Function] = []
    seen_funcs: set[base.Function] = set()
    
    # sorted order does not matter
    def add_func(f: base.Function):
        if f not in seen_funcs:
            funcs.append(f)
            seen_funcs.add(f)
    
    add_func(output.creator)
    txt += _dot_var(output, verbose)
    
    while funcs:
        f = funcs.pop()
        txt += _dot_func(f)
        for x in f.inputs:
            txt += _dot_var(x, verbose)
            if x.creator is not None:
                add_func(x.creator)
    return f"digraph g {{\n{txt}}}"

def plot_dot_graph(output: base.Variable, verbose=True, to_file="graph.png"):
    dot_graph = get_dot_graph(output, verbose)

    # save dot data to a temporary file
    with tempfile.TemporaryDirectory() as tmp_dir:
        graph_path = os.path.join(tmp_dir, "tmp_graph.dot")
        
        with open(graph_path, "w") as f:
            f.write(dot_graph)
            
        # generate graph image
        extension = os.path.splitext(to_file)[1][1:]
        cmd = f"dot {graph_path} -T {extension} -o {to_file}"
        subprocess.run(cmd, shell=True)
    
    # jupyter
    try:
        from IPython import display
        return display.Image(filename=to_file)
    except ImportError:
        pass
    
# =============================================================================
# Utility functions for numpy
# =============================================================================
def reshape_sum_backward(
    gy: base.Variable, x_shape: tuple[int], axis: Union[int, tuple[int, ...], None], keepdims: bool
) -> base.Variable:
    """Make the shape of the gradient array match the shape of the input array. (broadcastable)
    """
    ndim = len(x_shape)

    if axis is None:
        tupled_axis = None
    elif not isinstance(axis, tuple):
        tupled_axis = (axis,)
    else:
        tupled_axis = axis
    
    if not (ndim == 0 or tupled_axis is None or keepdims):
        # insert dimensions to match the input array
        actual_axis = [a if a >= 0 else a + ndim for a in tupled_axis] # convert negative axis to their positive equivalents
        shape = list(gy.shape)
        for a in sorted(actual_axis):
            shape.insert(a, 1)
    else:
        # no need to insert dimensions
        shape = gy.shape
    gy = gy.reshape(shape)
    return gy

def sum_to(x: NDArray, shape: tuple[int, ...]) -> NDArray:
    ndim = len(shape)
    lead = x.ndim - ndim
    lead_axis = tuple(range(lead))

    axis = tuple([i + lead for i, sx in enumerate(shape) if sx == 1])
    y = x.sum(lead_axis + axis, keepdims=True)
    if lead > 0:
        y = y.squeeze(lead_axis)
    return y
