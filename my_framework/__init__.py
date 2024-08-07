# is_simple_core = False

# if is_simple_core:
#     from my_framework.core_simple import (
#         Variable,
#         Function,
#         using_config,
#         no_grad,
#         as_array,
#         as_variable,
#     )
# else:
from my_framework.core import (
    Variable,
    Function,
    using_config,
    no_grad,
    as_array,
    as_variable,
)

import my_framework.utils
import my_framework.cuda
import my_framework.functions