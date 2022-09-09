#ifndef _FLEXFLOW_SOFTMAX_H
#define _FLEXFLOW_SOFTMAX_H

#include "flexflow/model.h"

namespace FlexFlow {

class Softmax;

class SoftmaxMeta : public OpMeta {
public:
  SoftmaxMeta(FFHandler handle,
              Softmax const *softmax,
              Legion::Domain const &input_domain);
#if defined(FF_USE_CUDA) || defined(FF_USE_HIP_CUDA)
  cudnnTensorDescriptor_t inputTensor;
#else
  miopenTensorDescriptor_t inputTensor;
#endif
  bool profiling;
  int dim;
  char op_name[MAX_OPNAME];
};

class Softmax : public Op {
public:
  Softmax(FFModel &model,
          const ParallelTensor logit,
          int dim,
          char const *name);
  void init(FFModel const &) override;
  void forward(FFModel const &) override;
  void backward(FFModel const &) override;
  void reset_idx(FFModel const &) override;
  void pipeinit(FFModel const &) override;
  void pipeforward(FFModel const &) override;
  void pipebackward(FFModel const &) override;
  bool get_int_parameter(PMParameter, int *) const override;
  void print_layer(FFModel const &model) override {
    assert(0);
  }
  static Op *
      create_operator_from_layer(FFModel &model,
                                 Layer const *layer,
                                 std::vector<ParallelTensor> const &inputs);
  static OpMeta *init_task(Legion::Task const *task,
                           std::vector<Legion::PhysicalRegion> const &regions,
                           Legion::Context ctx,
                           Legion::Runtime *runtime);
  static void forward_task(Legion::Task const *task,
                           std::vector<Legion::PhysicalRegion> const &regions,
                           Legion::Context ctx,
                           Legion::Runtime *runtime);
  static void backward_task(Legion::Task const *task,
                            std::vector<Legion::PhysicalRegion> const &regions,
                            Legion::Context ctx,
                            Legion::Runtime *runtime);
  void init_meta(SoftmaxMeta *m,
                 Legion::Rect<2> const &input,
                 Legion::Rect<2> const &output) const;
  bool measure_operator_cost(Simulator *sim,
                             MachineView const &pc,
                             CostMetrics &cost_metrics) const override;
  static void forward_kernel(SoftmaxMeta const *m,
                             float const *input_ptr,
                             float *output_ptr,
                             ffStream_t stream);
  static void forward_kernel_wrapper(SoftmaxMeta const *m,
                                     float const *input_ptr,
                                     float *output_ptr);
  static void backward_kernel(float *input_grad_ptr,
                              float const *output_grad_ptr,
                              size_t num_elements,
                              ffStream_t stream);
  static void backward_kernel_wrapper(SoftmaxMeta const *m,
                                      float *input_grad_ptr,
                                      float const *output_grad_ptr,
                                      size_t num_elements);
  size_t get_params_hash() const override;

private:
  template <int NDIM>
  static void
      forward_task_with_dim(Legion::Task const *task,
                            std::vector<Legion::PhysicalRegion> const &regions,
                            Legion::Context ctx,
                            Legion::Runtime *runtime);
  template <int NDIM>
  static void
      backward_task_with_dim(Legion::Task const *task,
                             std::vector<Legion::PhysicalRegion> const &regions,
                             Legion::Context ctx,
                             Legion::Runtime *runtime);

public:
  int dim;
  int fwd_input_idx = 0;
  int bwd_input_idx = 0;
  int fwd_output_idx = 0;
  int bwd_output_idx = 0;
  int init_output_idx = 0;
};

}; // namespace FlexFlow

#endif // _FLEXFLOW_SOFTMAX_H
