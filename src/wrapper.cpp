#include <algorithm>
#include <fstream>
#include <memory>
#include <utility>
#include <vector>

#include <ffm.h>

#include "common.h"
#include "vendor.h"

namespace
{
    // Modified from ffm.cpp txt2bin, must guarantee they write the same format.
    // Instead of parsing values from file, reads from dense y array and sparse
    // x array.
    // Fields are passed in as extra array of dimension x_cols.
    template <typename T, typename U>
    void raw_arr2bin(int32_t x_rows, int32_t x_cols, const T* y_arr,
                     const int32_t* x_field, const U* x_data,
                     const int32_t* x_indices, const int32_t* x_indptr,
                     const std::string& bin_path)
    {
        std::ofstream f_bin(bin_path, std::ios::out | std::ios::binary);

        std::vector<char> line(vendor::kMaxLineSize);

        ffm::ffm_long p = 0;
        vendor::disk_problem_meta meta;

        std::vector<ffm::ffm_float> Y;
        std::vector<ffm::ffm_float> R;
        std::vector<ffm::ffm_long> P(1, 0);
        std::vector<ffm::ffm_node> X;
        std::vector<ffm::ffm_long> B;

        auto write_chunk = [&]()
        {
            B.push_back(f_bin.tellp());
            ffm::ffm_int l = Y.size();
            ffm::ffm_long nnz = P[l];
            meta.l += l;

            f_bin.write(reinterpret_cast<char*>(&l), sizeof(ffm::ffm_int));
            f_bin.write(reinterpret_cast<char*>(Y.data()),
                        sizeof(ffm::ffm_float) * l);
            f_bin.write(reinterpret_cast<char*>(R.data()),
                        sizeof(ffm::ffm_float) * l);
            f_bin.write(reinterpret_cast<char*>(P.data()),
                        sizeof(ffm::ffm_long) * (l + 1));
            f_bin.write(reinterpret_cast<char*>(X.data()),
                        sizeof(ffm::ffm_node) * nnz);

            Y.clear();
            R.clear();
            P.assign(1, 0);
            X.clear();
            p = 0;
            meta.num_blocks++;
        };

        f_bin.write(reinterpret_cast<char*>(&meta),
                    sizeof(vendor::disk_problem_meta));

        for(int32_t i = 0; i < x_rows; i++)
        {
            ffm::ffm_float y = y_arr[i] > 0 ? 1.0f : -1.0f;

            ffm::ffm_float scale = 0;
            auto x_begin = x_indptr[i];
            auto x_end = x_indptr[i + 1];
            for(auto j = x_begin; j < x_end; j++, p++)
            {
                auto col = x_indices[j];

                ffm::ffm_node N;
                N.f = x_field[col];
                N.j = col;
                N.v = x_data[j];

                X.push_back(N);

                meta.m = std::max(meta.m, N.f + 1);
                meta.n = std::max(meta.n, N.j + 1);

                scale += N.v * N.v;
            }
            scale = 1.0 / scale;

            Y.push_back(y);
            R.push_back(scale);
            P.push_back(p);

            if(X.size() > (size_t)vendor::kCHUNK_SIZE)
                write_chunk();
        }
        write_chunk();
        write_chunk(); // write a dummy empty chunk in order to know where the
                       // EOF is
        assert(meta.num_blocks == (ffm::ffm_int)B.size());
        meta.B_pos = f_bin.tellp();
        f_bin.write(reinterpret_cast<char*>(B.data()),
                    sizeof(ffm::ffm_long) * B.size());

        meta.hash1 = 0xbadface; // dummy hash values
        meta.hash2 = 0xbadbeef;

        f_bin.seekp(0, std::ios::beg);
        f_bin.write(reinterpret_cast<char*>(&meta),
                    sizeof(vendor::disk_problem_meta));
    }

    template <typename T, typename U>
    void arr2bin(int32_t x_rows, int32_t x_cols, nc_array_t<T> y_arr,
                 nc_array_t<int32_t> x_field, nc_array_t<U> x_data,
                 nc_array_t<int32_t> x_indices, nc_array_t<int32_t> x_indptr,
                 const std::string& bin_path)
    {
        return raw_arr2bin(x_rows, x_cols, y_arr.data(), x_field.data(),
                           x_data.data(), x_indices.data(), x_indptr.data(),
                           bin_path);
    }

    // Transfer ownership of ptr to an array.
    // deleter is called when there are no more references to the memory.
    template <typename T>
    nc_array_t<T> as_array(int32_t size, T* ptr, void (*deleter)(void*))
    {
        auto capsule = py::capsule(ptr, deleter);
        return nc_array_t<T>{size, ptr, capsule};
    }

    // Transfer ownership of model weights to an array and return model fields
    // as tuple.
    auto as_tuple(ffm::ffm_model& model)
    {
        auto size = vendor::get_w_size(model);
        auto weights = model.W;
        model.W = nullptr; // prevent deletion in destructor

#ifdef _WIN32
        auto deleter = [](void* ptr) { _aligned_free(ptr); };
#else
        auto deleter = [](void* ptr) { std::free(ptr); };
#endif

        return std::tuple{model.n, model.m, model.k,
                          as_array(size, weights, deleter),
                          model.normalization};
    }

    auto train_on_disk(std::string train_path, std::string validation_path,
                       float eta, float lambda, int nr_iters, int k,
                       bool normalization, bool auto_stop)
    {
        auto model = ffm::ffm_train_on_disk(std::move(train_path),
                                            std::move(validation_path),
                                            {
                                                eta,
                                                lambda,
                                                nr_iters,
                                                k,
                                                normalization,
                                                auto_stop,
                                            });
        return as_tuple(model);
    }

    void save_model(int n, int m, int k, nc_array_t<ffm::ffm_float> weights,
                    bool normalization, std::string path)
    {
        auto model = ffm::ffm_model{
            n, m, k, weights.mutable_data(), normalization,
        };
        ffm::ffm_save_model(model, std::move(path));
        model.W = nullptr; // prevent deletion in destructor
    }

    auto load_model(std::string path)
    {
        auto model = ffm::ffm_load_model(std::move(path));
        return as_tuple(model);
    }

    template <typename T>
    nc_array_t<float> raw_predict(int n, int m, int k, ffm::ffm_float* weights,
                                  bool normalization, int32_t x_rows,
                                  int32_t x_cols, const int32_t* x_field,
                                  const T* x_data, const int32_t* x_indices,
                                  const int32_t* x_indptr)
    {
        auto model = ffm::ffm_model{
            n, m, k, weights, normalization,
        };

        nc_array_t<float> preds(x_rows);
        auto preds_ptr = preds.mutable_data();
        for(int32_t i = 0; i < x_rows; i++)
        {
            auto x_begin = x_indptr[i];
            auto x_end = x_indptr[i + 1];
            std::vector<ffm::ffm_node> nodes;
            nodes.reserve(x_begin - x_end);
            for(auto j = x_begin; j < x_end; j++)
            {
                auto col = x_indices[j];
                nodes.push_back(ffm::ffm_node{x_field[col], col,
                                              ffm::ffm_float(x_data[j])});
            }
            auto pred = ffm::ffm_predict(nodes.data(),
                                         nodes.data() + nodes.size(), model);
            preds_ptr[i] = pred;
        }

        model.W = nullptr; // prevent deletion in destructor
        return preds;
    }

    template <typename T>
    nc_array_t<float>
    predict(int n, int m, int k, nc_array_t<ffm::ffm_float> weights,
            bool normalization, int32_t x_rows, int32_t x_cols,
            nc_array_t<int32_t> x_field, nc_array_t<T> x_data,
            nc_array_t<int32_t> x_indices, nc_array_t<int32_t> x_indptr)
    {
        return raw_predict(n, m, k, weights.mutable_data(), normalization,
                           x_rows, x_cols, x_field.data(), x_data.data(),
                           x_indices.data(), x_indptr.data());
    }

} // namespace

PYBIND11_MODULE(wrapper, m)
{
    for_fundamental_types(
        [&](auto type)
        {
            using T = typename decltype(type)::type;
            m.def("arr2bin", &arr2bin<T, float>,
                  "convert arrays to binary ffm data files");
            m.def("arr2bin", &arr2bin<T, double>,
                  "convert arrays to binary ffm data files");
            m.def("predict", &predict<T>, "predict with model");
        });

    m.def("train_on_disk", &train_on_disk, "train with binary ffm data files");
    m.def("save_model", &save_model, "save model to file");
    m.def("load_model", &load_model, "load model from file");
}