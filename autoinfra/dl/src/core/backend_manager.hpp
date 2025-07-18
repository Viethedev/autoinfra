#pragma once
#include "core/backend.hpp"
#include <vector>
#include <mutex>

namespace core {

class BackendManager {
public:
    static BackendManager& instance() {
        static BackendManager mgr;
        return mgr;
    }

    void register_backend(BackendPtr backend) {
        std::lock_guard<std::mutex> lock(mutex_);
        backends_.push_back(std::move(backend));
    }

    const std::vector<BackendPtr>& backends() const {
        return backends_;
    }

    std::vector<DevicePtr> all_devices() const {
        std::vector<DevicePtr> all;
        for (const auto& backend : backends()) {
            const auto& devs = backend->devices();
            all.insert(all.end(), devs.begin(), devs.end());
        }
        return all;
    }

    DevicePtr get_device(DeviceType type, int index) const {
        for (const auto& backend : backends_) {
            for (const auto& dev : backend->devices()) {
                if (dev->type() == type && dev->index() == index) {
                    return dev;
                }
            }
        }
        throw std::runtime_error("Device not found");
    }

private:
    BackendManager() = default;

    std::vector<BackendPtr> backends_;
    mutable std::mutex mutex_;
};

} // namespace core
