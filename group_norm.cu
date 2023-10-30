#include "common.hpp"
#include <CL/sycl.hpp>

#define AlignMent 4096
#define ITER 1

// This funciton is to back compatible with old DPCPP compiler
template <typename T, int Dims = 1>
inline T* ITEXGetLocalAccPointer(const sycl::local_accessor<T, Dims> &accessor){
  if constexpr (std::is_same_v<decltype(accessor.get_pointer()), sycl::local_ptr<T>>){
    return accessor.get_pointer().get();
  }else{
    return accessor.get_pointer();
  }
}

template <typename T> using LocalAcc = sycl::local_accessor<T, 1>;

using OpKernelContext = sycl::queue;
// --------------------// GroupMeanVar //-------------------- //

template <int SUB_GROUP_SIZE, typename T>
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

template <int SUB_GROUP_SIZE, typename T, typename U, int VECSize>
struct PartialSumKernel {
  PartialSumKernel(const T *input, LocalAcc<U> scratch, U *temp_sum,
                   U *temp_sqr, const InputShape &shape, int scaled_hw)
      : input_(input), scratch_(scratch), temp_sum_(temp_sum),
        temp_sqr_(temp_sqr), num_hw_(shape.num_hw),
        num_channels_(shape.num_channels), num_groups_(shape.num_groups),
        chans_per_group_(shape.chans_per_group), scaled_hw_(scaled_hw) {}

  [[intel::reqd_sub_group_size(SUB_GROUP_SIZE)]] void
  operator()(sycl::nd_item<2> item) const {
    // TODO: consider case, that channs_per_group % VECSize != 0
    int batch_id = item.get_group(0);
    int scaled_id = item.get_group(1);
    int group_size = item.get_local_range(1);
    int id = item.get_local_id(1);

    auto sg = item.get_sub_group();
    int sg_id = sg.get_group_id();
    int sg_local_id = sg.get_local_id();
    int num_sg = sg.get_group_linear_range();

    const T *p_input = input_ + batch_id * num_hw_ * num_channels_;
    U *lmem = ITEXGetLocalAccPointer(scratch_);

    for (int idx = id * VECSize; idx < num_channels_;
         idx += group_size * VECSize) {
      sycl::vec<U, VECSize> sum{0};
      sycl::vec<U, VECSize> sqr{0};

      for (int ihw = scaled_id; ihw < num_hw_; ihw += scaled_hw_) {
        sycl::vec<T, VECSize> value =
            *(reinterpret_cast<const sycl::vec<T, VECSize> *>(
                p_input + ihw * num_channels_ + idx));

        for (int j = 0; j < VECSize; ++j) {
          U acc = static_cast<U>(value[j]);
          sum[j] += acc;
          sqr[j] += acc * acc;
        }
      }

      *(reinterpret_cast<sycl::vec<U, VECSize> *>(lmem + idx)) = sum;
      *(reinterpret_cast<sycl::vec<U, VECSize> *>(lmem + idx + num_channels_)) =
          sqr;
    }
    item.barrier(sycl::access::fence_space::local_space);

    for (int group_id = sg_id; group_id < num_groups_; group_id += num_sg) {
      U *data = lmem + group_id * chans_per_group_;

      U sum = U(0);
      U sqr = U(0);
      for (int i = sg_local_id; i < chans_per_group_; i += SUB_GROUP_SIZE) {
        sum += data[i];
        sqr += data[i + num_channels_];
      }

#pragma unroll
      for (int s = SUB_GROUP_SIZE >> 1; s > 0; s >>= 1) {
        sum += sycl::shift_group_left(sg, sum, s);
        sqr += sycl::shift_group_left(sg, sqr, s);
      }

      if (sg_local_id == 0) {
        int offset = batch_id * scaled_hw_ * num_groups_ +
                     group_id * scaled_hw_ + scaled_id;
        temp_sum_[offset] = sum;
        temp_sqr_[offset] = sqr;
      }
    }
  }

private:
  const T *input_;
  LocalAcc<U> scratch_;
  U *temp_sum_;
  U *temp_sqr_;
  int num_hw_;
  int num_channels_;
  int num_groups_;
  int chans_per_group_;
  int scaled_hw_;
};

// Compute sum and square sum of data in each group
template <int SUB_GROUP_SIZE, typename T, typename U>
sycl::event LaunchPartialSumKernel(sycl::queue *queue, const T *input,
                                   U *temp_sum, U *temp_sqr,
                                   const InputShape &shape, int scaled_hw) {
  size_t max_group_size =
      (*queue).get_device().get_info<sycl::info::device::max_work_group_size>();

  int VECSize = 4;
  while (shape.chans_per_group % VECSize != 0 && VECSize > 1) {
    VECSize >>= 1;
  }

  size_t group_size = SUB_GROUP_SIZE;
  while (group_size << 1 <= (shape.num_channels / VECSize))
    group_size <<= 1;
  group_size = std::min(group_size, max_group_size);

  // shared local memory size
  size_t lmem_size = shape.num_channels << 1;

  // Create the range object
  sycl::range<2> global(shape.num_batches, scaled_hw * group_size);
  sycl::range<2> local(1, group_size);
  sycl::nd_range<2> range(global, local);

  sycl::event evt;
  if  (VECSize == 4) {                                                                     
    queue->submit([&](sycl::handler &cgh) {                              
      LocalAcc<U> scratch(sycl::range<1>{lmem_size}, cgh);                     
      PartialSumKernel<SUB_GROUP_SIZE, T, U, 4> task(                          
          input, scratch, temp_sum, temp_sqr, shape, scaled_hw);               
      cgh.parallel_for<PartialSumKernel<SUB_GROUP_SIZE, T, U, 4>>(range,       
                                                                  task);       
    });  
    return evt;                                                                      
  } else if  (VECSize == 2){                                                                    
    queue->submit([&](sycl::handler &cgh) {                              
      LocalAcc<U> scratch(sycl::range<1>{lmem_size}, cgh);                     
      PartialSumKernel<SUB_GROUP_SIZE, T, U, 2> task(                          
          input, scratch, temp_sum, temp_sqr, shape, scaled_hw);               
      cgh.parallel_for<PartialSumKernel<SUB_GROUP_SIZE, T, U, 2>>(range,       
                                                                  task);       
    });  
    return evt;                                                                      
  }else if (VECSize == 1){                                                                    
    queue->submit([&](sycl::handler &cgh) {                              
      LocalAcc<U> scratch(sycl::range<1>{lmem_size}, cgh);                     
      PartialSumKernel<SUB_GROUP_SIZE, T, U, 1> task(                          
          input, scratch, temp_sum, temp_sqr, shape, scaled_hw);               
      cgh.parallel_for<PartialSumKernel<SUB_GROUP_SIZE, T, U, 1>>(range,       
                                                                  task);       
    });                                                                                
    return evt;
  }else{
    std::cout << "error: " << VECSize << std::endl;
    abort();
  }
  
}

template <int SUB_GROUP_SIZE, typename T> struct MeanFromPartialKernel {
  MeanFromPartialKernel(const T *temp_sum, const T *temp_sqr,
                        LocalAcc<T> scratch, T *temp_mean, T *temp_var,
                        const InputShape &shape, int scaled_hw)
      : temp_sum_(temp_sum), temp_sqr_(temp_sqr), scratch_(scratch),
        temp_mean_(temp_mean), temp_var_(temp_var), num_hw_(shape.num_hw),
        num_channels_(shape.num_channels), num_groups_(shape.num_groups),
        chans_per_group_(shape.chans_per_group), scaled_hw_(scaled_hw) {}

  [[intel::reqd_sub_group_size(SUB_GROUP_SIZE)]] void
  operator()(sycl::nd_item<2> item) const {
    int batch_id = item.get_group(0);
    int group_id = item.get_group(1);
    int group_size = item.get_local_range(1);
    int id = item.get_local_id(1);

    int offset = batch_id * scaled_hw_ * num_groups_ + group_id * scaled_hw_;
    const T *p_sum = temp_sum_ + offset;
    const T *p_sqr = temp_sqr_ + offset;

    T sum = 0;
    T sqr = 0;
    for (int i = id; i < scaled_hw_; i += group_size) {
      sum += p_sum[i];
      sqr += p_sqr[i];
    }

    T *lmem = ITEXGetLocalAccPointer(scratch_);
    int total = num_hw_ * chans_per_group_;
    GroupMeanVar<SUB_GROUP_SIZE>(item, &sum, &sqr, total, lmem);

    if (id == 0) {
      int offset = batch_id * num_groups_ + group_id;
      temp_mean_[offset] = sum;
      temp_var_[offset] = sqr;
    }
  }

private:
  const T *temp_sum_;
  const T *temp_sqr_;
  LocalAcc<T> scratch_;
  T *temp_mean_;
  T *temp_var_;
  int num_hw_;
  int num_channels_;
  int num_groups_;
  int chans_per_group_;
  int scaled_hw_;
};

// Compute sum and square sum of data in each group
template <int SUB_GROUP_SIZE, typename T>
sycl::event LaunchMeanFromPartialKernel(sycl::queue *queue, const T *temp_sum,
                                        const T *temp_sqr, T *temp_mean,
                                        T *temp_var, const InputShape &shape,
                                        int scaled_hw) {
  auto max_group_size =
      (*queue).get_device().get_info<sycl::info::device::max_work_group_size>();

  size_t group_size = SUB_GROUP_SIZE;
  while (group_size << 1 <= scaled_hw)
    group_size <<= 1;
  group_size = std::min(group_size, max_group_size);

  // shared local memory size
  size_t lmem_size = group_size / SUB_GROUP_SIZE * 2;

  // Create the range object
  sycl::range<2> global(shape.num_batches, shape.num_groups * group_size);
  sycl::range<2> local(1, group_size);
  sycl::nd_range<2> range(global, local);

  sycl::event evt;
  queue->submit([&](sycl::handler &cgh) {
    LocalAcc<T> scratch(sycl::range<1>{lmem_size}, cgh);
    MeanFromPartialKernel<SUB_GROUP_SIZE, T> task(
        temp_sum, temp_sqr, scratch, temp_mean, temp_var, shape, scaled_hw);
    cgh.parallel_for<MeanFromPartialKernel<SUB_GROUP_SIZE, T>>(range, task);
  });
  return evt;
}

template <typename T, typename U>
void ComputeMeanAndVar(OpKernelContext *queue, const T *input, U *temp_mean,
                       U *temp_var, const InputShape &shape) {
  size_t bytes =
      shape.num_batches * shape.num_channels * shape.num_hw * sizeof(T) +
      shape.num_batches * shape.num_groups * sizeof(U) * 2;
  int num_to_reduce = shape.num_hw * shape.chans_per_group;
  //  As we use one workgroup to each output elems
  bool use_one_kernel = (num_to_reduce < 64 * 1024);
  std::cout << "use one kernel: " << use_one_kernel << std::endl;
  if (use_one_kernel) {
    sycl::event event;
    if (shape.chans_per_group < 32) {
      event =
          LaunchMeanAndVarKernel<16>(queue, input, temp_mean, temp_var, shape);
    } else {
      event =
          LaunchMeanAndVarKernel<32>(queue, input, temp_mean, temp_var, shape);
    }
    float duration = 1e-3; // get_exe_time(event); // us
    float bw = bytes / 1e3 / duration;
    std::cout << "bytes: " << bytes / 1024 / 1024 << " Mb, time: " << duration
              << " us, bandwidth: " << bw << " GB/s" << std::endl;
  } else {
    int scaled_hw = std::min(512, shape.num_hw);

    // allocate temporary for sum and square sum
    int scratch_shape = 2 * shape.num_batches * shape.num_groups * scaled_hw;

    U *temp_sum = sycl::malloc_device<U>(scratch_shape, *queue);
    U *temp_sqr = temp_sum + shape.num_batches * shape.num_groups * scaled_hw;

    sycl::event evt1, evt2;
    if (shape.chans_per_group < 32) {
      evt1 = LaunchPartialSumKernel<16>(queue, input, temp_sum, temp_sqr, shape,
                                        scaled_hw);
    } else {
      evt1 = LaunchPartialSumKernel<32>(queue, input, temp_sum, temp_sqr, shape,
                                        scaled_hw);
    }
    if (shape.chans_per_group < 32) {
      evt2 = LaunchMeanFromPartialKernel<16>(
          queue, temp_sum, temp_sqr, temp_mean, temp_var, shape, scaled_hw);
    } else {
      evt2 = LaunchMeanFromPartialKernel<32>(
          queue, temp_sum, temp_sqr, temp_mean, temp_var, shape, scaled_hw);
    }
    float duration1 = 1e-4; // get_exe_time(evt1);
    float duration2 = 1e-4; // get_exe_time(evt2);
    float bw = bytes / 1e3 / (duration1 + duration2);
    std::cout << "kernel1: " << duration1 << " us, kernel2: " << duration2
              << " us" << std::endl;
    std::cout << "bytes: " << bytes / 1024 / 1024
              << " Mb, time: " << (duration1 + duration2)
              << " us, bandwidth: " << bw << " GB/s" << std::endl;
  }
}

template <typename T, typename U>
void verifyMeanAndVar(const T *in, const U *mean, const U *var,
                      const InputShape &shape) {
  int mean_size = shape.num_batches * shape.num_groups;
  std::vector<U> ref_mean(mean_size);
  std::vector<U> ref_var(mean_size);

  for (int bs = 0; bs < shape.num_batches; ++bs) {
    for (int g = 0; g < shape.num_groups; ++g) {
      U sum = U(0);
      U sum_of_square = U(0);
      for (int hw = 0; hw < shape.num_hw; ++hw) {
        for (int cg = 0; cg < shape.chans_per_group; ++cg) {
          int offset = bs * shape.num_hw * shape.num_channels +
                       g * shape.chans_per_group + hw * shape.num_channels + cg;
          sum += in[offset];
          sum_of_square += (in[offset] * in[offset]);
        }
      }
      ref_mean[bs * shape.num_groups + g] =
          sum / (shape.num_hw * shape.chans_per_group);
      ref_var[bs * shape.num_groups + g] =
          sum_of_square / (shape.num_hw * shape.chans_per_group) -
          ref_mean[bs * shape.num_groups + g] *
              ref_mean[bs * shape.num_groups + g];
    }
  }
  float atol = 1e-5;
  float rtol = 1e-5;
  if (std::is_same<T, sycl::half>::value) {
    atol = 1e-3;
    rtol = 1e-3;
  }

  if (all_closer(ref_mean.data(), mean, mean_size, atol, rtol)) {
    std::cout << "mean passed" << std::endl;
  } else {
    std::cout << "mean failed" << std::endl;
  }

  if (all_closer(ref_var.data(), var, mean_size, atol, rtol)) {
    std::cout << "var passed" << std::endl;
  } else {
    std::cout << "var failed" << std::endl;
  }
}

int main(int argc, const char **argv) {
  typedef sycl::half T;
  typedef float U;
  sycl::queue queue(
      sycl::gpu_selector_v,
      sycl::property_list{sycl::property::queue::in_order()});
  int bs = 2;
  int ic = 32 * 16;
  int ih = 512;
  int iw = 512;
  if (argc > 4) {
    bs = std::stoi(argv[1]);
    ih = std::stoi(argv[2]);
    iw = std::stoi(argv[3]);
    ic = std::stoi(argv[4]);
  }
  std::cout << "bs * ih * iw * ic: " << bs << " * " << ih << " * " << iw
            << " * " << ic << std::endl;

  const int groups = 32;
  int size = bs * ic * ih * iw;
  int mean_size = bs * groups;
  InputShape shape{bs, ih * iw, ic, groups, ic / groups};

  T *in =
      sycl::aligned_alloc_shared<T>(AlignMent, size, queue); // (bs, ih, iw, ic)
  U *mean = sycl::aligned_alloc_shared<U>(AlignMent, mean_size, queue);
  U *var = sycl::aligned_alloc_shared<U>(AlignMent, mean_size, queue);

  ComputeMeanAndVar(&queue, in, mean, var, shape);

  for (int i = 0; i < ITER; ++i) {
    init_random(in, size);
    ComputeMeanAndVar(&queue, in, mean, var, shape);
  }

  queue.wait();
  verifyMeanAndVar(in, mean, var, shape);

  sycl::free(in, queue);
  sycl::free(mean, queue);
  sycl::free(var, queue);
}
