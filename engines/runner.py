"""Runs a serialized TensorRT engine using TensorRT 10+'s tensor-address
execution API. Input and output buffers are plain CUDA torch.Tensors;
TensorRT only needs their device addresses (set_tensor_address), and a
torch CUDA stream drives execute_async_v3 so timing can share the same
stream a real serving process would use.

Buffer dtypes are read from the engine itself (get_tensor_dtype), not
assumed to be float32: the fp16 engine's graph is fp16 end to end (see
export_fp16.py), so its input and output tensor-address buffers must be
fp16, or TensorRT's set_tensor_address silently reads/writes past what
the caller allocated. infer() always returns float32, converting as
needed, since every caller compares logits across precisions and a
mixed-dtype comparison would be its own source of spurious diffs.
"""
import tensorrt as trt
import torch

_TRT_TO_TORCH = {
    trt.DataType.FLOAT: torch.float32,
    trt.DataType.HALF: torch.float16,
    trt.DataType.INT8: torch.int8,
}

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


class EngineRunner:
    def __init__(self, engine_path: str, max_batch: int, num_classes: int = 10):
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(TRT_LOGGER)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.input_name = self.engine.get_tensor_name(0)
        self.output_name = self.engine.get_tensor_name(1)
        self.input_dtype = _TRT_TO_TORCH[self.engine.get_tensor_dtype(self.input_name)]
        self.output_dtype = _TRT_TO_TORCH[self.engine.get_tensor_dtype(self.output_name)]
        self.max_batch = max_batch
        self.num_classes = num_classes
        self.stream = torch.cuda.Stream()

    def infer(self, images: torch.Tensor) -> torch.Tensor:
        """images: CUDA tensor (B, 3, 32, 32), any float dtype, B <= max_batch.
        Always returns float32 logits."""
        batch = images.shape[0]
        if images.dtype != self.input_dtype:
            images = images.to(self.input_dtype)
        images = images.contiguous()
        self.context.set_input_shape(self.input_name, tuple(images.shape))
        out = torch.empty((batch, self.num_classes), dtype=self.output_dtype, device="cuda")

        self.context.set_tensor_address(self.input_name, images.data_ptr())
        self.context.set_tensor_address(self.output_name, out.data_ptr())

        with torch.cuda.stream(self.stream):
            self.context.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()
        return out.float()
