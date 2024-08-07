is_simple_core = False

if is_simple_core:
    from core.core_simple import (
        Variable,
        Function,
        using_config,
        no_grad,
        as_array,
        as_variable,
    )
else:
    from core.core import (
        Variable,
        Function,
        using_config,
        no_grad,
        as_array,
        as_variable,
    )
