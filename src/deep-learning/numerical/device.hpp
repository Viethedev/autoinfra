namespace numerical
{

    enum class DeviceType
    {
        CPU,
        CUDA,
        ROCm,
        Vulkan,
        Metal
    };

    struct Device
    {
        bool is_gpu;
        int device_id;
        DeviceType device_type;

        static Device CPU() { return {false, 0, DeviceType::CPU}; }
        static Device GPU(int id, DeviceType type) { return {true, id, type}; }
    };

}
/* Use:
Device cpu = Device::CPU();
Device cuda_gpu0 = Device::GPU(0, GpuType::CUDA);
Device amd_gpu1  = Device::GPU(1, GpuType::ROCm);
*/