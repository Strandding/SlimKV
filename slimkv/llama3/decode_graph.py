"""Optional CUDA Graph replay for state-free, fixed-shape decode operations."""
import torch


class DecodeGraphs:
    def __init__(self):
        self.entries = {}

    def __call__(self, name, function, *inputs):
        if (torch.is_grad_enabled() or not inputs[0].is_cuda
                or torch.cuda.is_current_stream_capturing()):
            return function(*inputs)
        key = (name, tuple((tuple(x.shape), x.dtype, x.device, x.stride()) for x in inputs))
        if key not in self.entries:
            current = torch.cuda.current_stream(inputs[0].device)
            stream = torch.cuda.Stream(device=inputs[0].device)
            buffers = tuple(x.clone() for x in inputs)
            stream.wait_stream(current)
            with torch.cuda.stream(stream):
                for _ in range(3):
                    function(*buffers)
            current.wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                output = function(*buffers)
            current.wait_stream(stream)
            self.entries[key] = graph, buffers, output
        graph, buffers, output = self.entries[key]
        for target, source in zip(buffers, inputs):
            target.copy_(source)
        graph.replay()
        return output
