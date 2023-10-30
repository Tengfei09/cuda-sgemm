#include "common.hpp"
#include <CL/sycl.hpp>

#define AlignMent 4096
#define ITER 1

// --------------------// GroupMeanVar //-------------------- //

template <int WARP_SIZE, typename T>
void GroupMeanVar(const sycl::nd_item<2> &item, T *par_sum, T *par_sqr,
                  int total, T *lmem) {
  auto sg = item.get_sub_group();
  int sg_id = sg.get_group_id();
  int sg_local_id = sg.get_local_id();
  int num_sg = sg.get_group_linear_range();

  // compute sum of each sub group
  T sum = *par_sum;
  T sqr = *par_sqr;
#pragma unroll
  for (int s = SUB_GROUP_SIZE >> 1; s > 0; s >>= 1) {
    sum += sycl::shift_group_left(sg, sum, s);
    sqr += sycl::shift_group_left(sg, sqr, s);
  }

  if (sg_local_id == 0) {
    lmem[sg_id] = sum;
    lmem[sg_id + num_sg] = sqr;
  }
  item.barrier(sycl::access::fence_space::local_space);

  // compute total sum and mean by one sub group
  if (sg_id == 0) {
    sum = 0;
    sqr = 0;
    for (int i = sg_local_id; i < num_sg; i += SUB_GROUP_SIZE) {
      sum += lmem[i];
      sqr += lmem[i + num_sg];
    }

#pragma unroll
    for (int s = SUB_GROUP_SIZE >> 1; s > 0; s >>= 1) {
      if (s < num_sg) {
        sum += sycl::shift_group_left(sg, sum, s);
        sqr += sycl::shift_group_left(sg, sqr, s);
      }
    }

    sum = sum / total;
    sqr = sqr / total - sum * sum;
    (*par_sum) = sum;
    (*par_sqr) = sqr;
  }
}

template <int SUB_GROUP_SIZE, typename T, typename U> struct MeanAndVarKernel {
  MeanAndVarKernel(const T *input, LocalAcc<U> scratch, U *temp_mean,
                   U *temp_var, const InputShape &shape, int sx, int sy)
      : input_(input), scratch_(scratch), temp_mean_(temp_mean),
        temp_var_(temp_var), num_hw_(shape.num_hw),
        num_channels_(shape.num_channels),
        chans_per_group_(shape.chans_per_group), sx_(sx), sy_(sy) {}

  [[intel::reqd_sub_group_size(SUB_GROUP_SIZE)]] void
  operator()(sycl::nd_item<2> item) const {
    int batch_id = item.get_group(0);
    int group_id = item.get_group(1);
    int id = item.get_local_id(1);

    const T *p_input = input_ + batch_id * num_hw_ * num_channels_ +
                       group_id * chans_per_group_;

    int iy = id / sx_;
    int ix = id - iy * sx_;

    U sum = U(0);
    U sqr = U(0);
    for (int jy = iy; jy < num_hw_; jy += sy_) {
      const T *pval = p_input + jy * num_channels_;

      for (int jx = ix; jx < chans_per_group_; jx += sx_) {
        U value = static_cast<U>(pval[jx]);
        sum += value;
        sqr += value * value;
      }
    }

    U *lmem = ITEXGetLocalAccPointer(scratch_);
    int total = num_hw_ * chans_per_group_;
    GroupMeanVar<SUB_GROUP_SIZE>(item, &sum, &sqr, total, lmem);

    if (id == 0) {
      int offset = batch_id * (num_channels_ / chans_per_group_) + group_id;
      temp_mean_[offset] = sum;
      temp_var_[offset] = sqr;
    }
  }

private:
  const T *input_;
  LocalAcc<U> scratch_;
  U *temp_mean_;
  U *temp_var_;
  int num_hw_;
  int num_channels_;
  int chans_per_group_;
  int sx_;
  int sy_;
};

// Compute mean and variance in one kernel
template <int SUB_GROUP_SIZE, typename T, typename U>
sycl::event LaunchMeanAndVarKernel(sycl::queue *queue, const T *input,
                                   U *temp_mean, U *temp_var,
                                   const InputShape &shape) {
  int group_size =
      (*queue).get_device().get_info<sycl::info::device::max_work_group_size>();

  int sx = SUB_GROUP_SIZE;
  while (sx << 1 <= shape.chans_per_group)
    sx <<= 1;
  sx = std::min(sx, group_size);
  int sy = group_size / sx;

  // shared local memory size
  size_t lmem_size = group_size / SUB_GROUP_SIZE * 2;

  // Create the range object
  sycl::range<2> global(shape.num_batches, shape.num_groups * group_size);
  sycl::range<2> local(1, group_size);
  sycl::nd_range<2> range(global, local);

  sycl::event evt;
  queue->submit([&](sycl::handler &cgh) {
    LocalAcc<U> scratch(sycl::range<1>{lmem_size}, cgh);
    MeanAndVarKernel<SUB_GROUP_SIZE, T, U> task(input, scratch, temp_mean,
                                                temp_var, shape, sx, sy);
    cgh.parallel_for<MeanAndVarKernel<SUB_GROUP_SIZE, T, U>>(range, task);
  });
  return evt;
}
