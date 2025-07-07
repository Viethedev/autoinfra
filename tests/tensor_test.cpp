#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>
#include "tensor/tensor.hpp"

TEST_CASE("Tensor says hello", "[tensor]") {
    dl::Tensor t;
    t.hello();
    REQUIRE(true);
}
