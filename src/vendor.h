// vendor contains copies of libffm non-visible names which is used in pyffm.
#pragma once

#include <ffm.h>

namespace vendor
{
    struct disk_problem_meta
    {
        ffm::ffm_int n = 0;
        ffm::ffm_int m = 0;
        ffm::ffm_int l = 0;
        ffm::ffm_int num_blocks = 0;
        ffm::ffm_long B_pos = 0;
        uint64_t hash1;
        uint64_t hash2;
    };

    ffm::ffm_int const kALIGNByte = 16;

    ffm::ffm_int const kALIGN = kALIGNByte / sizeof(ffm::ffm_float);
    ffm::ffm_int const kCHUNK_SIZE = 10000000;
    ffm::ffm_int const kMaxLineSize = 100000;

    inline ffm::ffm_int get_k_aligned(ffm::ffm_int k)
    {
        return (ffm::ffm_int)ceil((ffm::ffm_float)k / kALIGN) * kALIGN;
    }

    inline ffm::ffm_long get_w_size(ffm::ffm_model& model)
    {
        ffm::ffm_int k_aligned = get_k_aligned(model.k);
        return (ffm::ffm_long)model.n * model.m * k_aligned * 2;
    }
} // namespace vendor
