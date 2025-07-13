
//✅ Check tensor device:

cpp
Copy
Edit
if (tensor->device().type() == DeviceType::CUDA) {
    std::cout << "Tensor is on CUDA device!" << std::endl;
}
//✅ Move between devices:

cpp
Copy
Edit
auto cpu_tensor = tensor->to(Device(DeviceType::CPU));
//✅ Logging:

cpp
Copy
Edit
std::cout << "Tensor is on device: " << tensor->device().to_string() << std::endl;