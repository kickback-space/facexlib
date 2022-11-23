import cv2
from PIL import Image
import numpy as np
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import threading


class TRTModel:
    def __init__(self, trt_engine_path, device=0):
        self.cfx = cuda.Device(device).make_context()
        stream = cuda.Stream()

        TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
        trt.init_libnvinfer_plugins(TRT_LOGGER, "")
        runtime = trt.Runtime(TRT_LOGGER)

        # deserialize engine
        with open(trt_engine_path, "rb") as f:
            buf = f.read()
            engine = runtime.deserialize_cuda_engine(buf)
        context = engine.create_execution_context()

        # prepare buffer
        host_inputs = []
        cuda_inputs = []
        host_outputs = []
        cuda_outputs = []
        input_shapes = []
        output_shapes = []
        bindings = []

        for binding in engine:
            shape = engine.get_binding_shape(binding)
            size = trt.volume(shape) * engine.max_batch_size
            host_mem = cuda.pagelocked_empty(size, np.float32)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(cuda_mem))
            if engine.binding_is_input(binding):
                host_inputs.append(host_mem)
                cuda_inputs.append(cuda_mem)
                input_shapes.append(shape)
            else:
                host_outputs.append(host_mem)
                cuda_outputs.append(cuda_mem)
                output_shapes.append(shape)
        # store
        self.stream = stream
        self.context = context
        self.engine = engine

        self.host_inputs = host_inputs
        self.cuda_inputs = cuda_inputs
        self.host_outputs = host_outputs
        self.cuda_outputs = cuda_outputs
        self.input_shapes = input_shapes
        self.output_shapes = output_shapes

        self.bindings = bindings

    def infer(self, inputs, host_outputs=[]):
        threading.Thread.__init__(self)
        self.cfx.push()

        # restore
        stream = self.stream
        context = self.context
        self.engine

        host_inputs = self.host_inputs
        cuda_inputs = self.cuda_inputs
        if len(host_outputs) == 0:
            host_outputs = self.host_outputs
        cuda_outputs = self.cuda_outputs
        bindings = self.bindings

        # read image
        for i, image in enumerate(inputs):
            np.copyto(host_inputs[i], image.ravel())
        # inference
        for i in range(len(host_inputs)):
            cuda.memcpy_htod_async(cuda_inputs[i], host_inputs[i], stream)
        context.execute_async(bindings=bindings, stream_handle=stream.handle)
        for i in range(len(host_outputs)):
            cuda.memcpy_dtoh_async(host_outputs[i], cuda_outputs[i], stream)
        stream.synchronize()

        # parse output
        outputs = []
        for output, output_shape in zip(host_outputs, self.output_shapes):
            output = output.reshape(output_shape)
            outputs.append(output)
        self.cfx.pop()
        return outputs

    def destroy(self):
        self.cfx.pop()
