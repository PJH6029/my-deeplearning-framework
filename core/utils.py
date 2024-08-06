from numpy.typing import NDArray
import os, subprocess, tempfile

import core

# =============================================================================
# visualization tools
# =============================================================================
def _dot_var(v: core.Variable, verbose=False) -> str:
    name = "" if v.name is None else v.name
    if verbose and v.data is not None:
        if v.name is not None:
            name += ": "
        name += f"{str(v.shape)} {str(v.dtype)}"
    return f'{id(v)} [label="{name}", color=orange, style=filled]\n'

def _dot_func(f: core.Function) -> str:
    txt = f"{id(f)} [label={f.__class__.__name__}, color=lightblue, style=filled, shape=box]\n"
    
    dot_edge = lambda v, w: f"{id(v)} -> {id(w)}\n"
    for x in f.inputs:
        txt += dot_edge(x, f)
    for y in f.outputs:
        txt += dot_edge(f, y()) # y is weakref
    return txt

def get_dot_graph(output: core.Variable, verbose=True) -> str:
    txt = ""
    funcs: list[core.Function] = []
    seen_funcs: set[core.Function] = set()
    
    # sorted order does not matter
    def add_func(f: core.Function):
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

def plot_dot_graph(output: core.Variable, verbose=True, to_file="graph.png"):
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