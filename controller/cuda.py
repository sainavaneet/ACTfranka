import pycuda.driver as cuda
cuda.init()

# Check the number of CUDA-capable devices
print("Detected {} CUDA Capable device(s).".format(cuda.Device.count()))

for i in range(cuda.Device.count()):
    gpu = cuda.Device(i)
    print("Device {}: {}".format(i, gpu.name()))
    compute_capability = f"{gpu.compute_capability_major}.{gpu.compute_capability_minor}"
    print("  Compute Capability: ", compute_capability)
    print("  Total Memory: {:.2f} GB".format(gpu.total_memory() / (1024 ** 3)))
