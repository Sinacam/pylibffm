#pragma once

#include <cstdint>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

using std::size_t;

using std::int16_t;
using std::int32_t;
using std::int64_t;
using std::int8_t;

using std::uint16_t;
using std::uint32_t;
using std::uint64_t;
using std::uint8_t;

template <typename T>
struct type_identity
{
    using type = T;
};

template <typename... Ts, typename F>
inline void for_each_type(F&& f)
{
    (f(type_identity<Ts>{}), ...);
}

template <typename F>
inline void for_fundamental_types(F&& f)
{
    for_each_type<int8_t, int16_t, int32_t, int64_t, uint8_t, uint16_t,
                  uint32_t, uint64_t, bool, float, double>(f);
}

template <typename>
struct dtype_num_t;

template <typename T>
inline constexpr int dtype_num_v = dtype_num_t<T>::value;

#define DEFINE_DTYPE_NUM_T(T, value)                                           \
    template <>                                                                \
    struct dtype_num_t<T> : std::integral_constant<int, value>                 \
    {                                                                          \
    };

DEFINE_DTYPE_NUM_T(bool, 0)
DEFINE_DTYPE_NUM_T(int8_t, 1)
DEFINE_DTYPE_NUM_T(int16_t, 3)
DEFINE_DTYPE_NUM_T(int32_t, 5)
DEFINE_DTYPE_NUM_T(int64_t, 7)
DEFINE_DTYPE_NUM_T(uint8_t, 2)
DEFINE_DTYPE_NUM_T(uint16_t, 4)
DEFINE_DTYPE_NUM_T(uint32_t, 6)
DEFINE_DTYPE_NUM_T(uint64_t, 8)
DEFINE_DTYPE_NUM_T(float, 11)
DEFINE_DTYPE_NUM_T(double, 12)

#undef DEFINE_DTYPE_NUM_T
