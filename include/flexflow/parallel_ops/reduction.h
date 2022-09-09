#ifndef _FLEXFLOW_REDUCTION_H
#define _FLEXFLOW_REDUCTION_H

#include "parallel_op.h"

namespace FlexFlow {

class Reduction : public ParallelOp {
public:
  Reduction(FFModel &model,
            const ParallelTensor input,
            int reduction_legion_dim,
            int reduction_degree,
            char const *name = NULL);

  void create_input_partition(FFModel &model) override;
  void init(FFModel const &) override;
  void forward(FFModel const &) override;
  void backward(FFModel const &) override;
  void reset_idx(FFModel const &) override {
    assert(0);
  }
  void pipeinit(FFModel const &) override {
    assert(0);
  }
  void pipeforward(FFModel const &) override {
    assert(0);
  }
  void pipebackward(FFModel const &) override {
    assert(0);
  }
  bool get_int_parameter(PMParameter, int *) const override;
  bool append_parallel_op_info(
      std::vector<ParallelOpInfo> &parallel_ops) const override;
  static void forward_task(Legion::Task const *task,
                           std::vector<Legion::PhysicalRegion> const &regions,
                           Legion::Context ctx,
                           Legion::Runtime *runtime);
  static void backward_task(Legion::Task const *task,
                            std::vector<Legion::PhysicalRegion> const &regions,
                            Legion::Context ctx,
                            Legion::Runtime *runtime);
  template <typename T>
  static void forward_kernel(const T *input_ptr,
                             T *output_ptr,
                             size_t num_elements,
                             size_t num_replicas);
  template <typename T>
  static void backward_kernel(const T *output_grad_ptr,
                              T *input_grad_ptr,
                              size_t num_elements);
  bool measure_operator_cost(Simulator *sim,
                             MachineView const &pc,
                             CostMetrics &cost_metrics) const override;

  size_t get_params_hash() const override;

public:
  int reduction_dim, reduction_degree;
};

}; // namespace FlexFlow

#endif // _FLEXFLOW_REDUCTION_H
