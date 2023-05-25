from .llava_process_function import (
    LLavaConvProcessV1,
    LlavaImageProcessorV1,
    LlavaTextProcessV1,
)

from .otter_process_function import (
    OtterConvProcess,
    OtterImageProcess,
    OtterTextProcess
)

from .box_process_function import (
    BoxFormatProcess,
    BoxFormatter,
    PlainBoxFormatter,
    smart_prepare_target_processor,
)
