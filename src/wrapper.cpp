#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

PYBIND11_MODULE(wrapper, m)
{
    m.def(
        "test", [](int x) { return 42; }, "test function");
}