/* Copyright 2020 Facebook
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "flexflow/ops/split.h"
#include "flexflow/utils/hash_utils.h"

namespace FlexFlow {
// declare Legion names
using Legion::Context;
using Legion::Runtime;
using Legion::Domain;
using Legion::Task;
using Legion::Rect;
using Legion::coord_t;
using Legion::PhysicalRegion;
using Legion::TaskLauncher;
using Legion::IndexLauncher;
using Legion::FutureMap;
using Legion::ArgumentMap;
using Legion::TaskArgument;
using Legion::RegionRequirement;
using Legion::Predicate;

void FFModel::split(const Tensor input,
                    Tensor* outputs,
                    const std::vector<int>& splits,
                    int axis,
                    const char* name)
{
  assert(false);
#ifdef DEADCODE
  Split* split = new Split(*this, input, splits, axis, name);
  layers.push_back(split);
  for (size_t i = 0; i < splits.size(); i++)
    outputs[i] = split->outputs[i];
#endif
}

size_t Split::get_params_hash() const {
  size_t hash = 0;
  for (int i = 0; i < this->numInputs; i++) {
    hash_combine(hash, this->inputs[i]->get_owner_independent_hash()); 
  }
  hash_combine(hash, this->axis);

  return hash;
}

Split::Split(FFModel& model,
             const ParallelTensor input,
             const std::vector<int>& splits,
             int _axis,
             const char* name)
: Op(model, OP_SPLIT, name, 1/*inputs*/, 0/*weights*/, splits.size()/*outputs*/, input),
  axis(input->num_dims-1-_axis)
{
  numOutputs = splits.size();
  // Note that we use the Legion dim ordering
  // axis = input->num_dims-1-_axis
  assert(axis >= 0);
  numWeights = 0;
  int split_size = 0;
  for (int i = 0; i < numOutputs; i++) {
    split_size += splits[i];
    int numdim = input->num_dims;
    ParallelDim dims[MAX_TENSOR_DIM];
    for (int j = 0; j < numdim; j++)
      dims[j] = input->dims[j];
    dims[axis].size = splits[i];
    // Assert the _axis dim cannot be parallelized
    assert(dims[axis].degree == 1);
    assert(dims[axis].parallel_idx == -1);
    outputs[i] = model.create_parallel_tensor_legion_ordering(
        numdim, dims, input->data_type,
        this/*owner_op*/, i/*owner_idx*/);
  }
  // Check split sizes
  assert(split_size == input->dims[axis].size);
}

void Split::init(const FFModel& ff)
{
  assert(check_output_input_weight_same_parallel_is());
  parallel_is = outputs[0]->parallel_is;
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  IndexLauncher launcher(SPLIT_INIT_TASK_ID, parallel_is,
                         TaskArgument(this, sizeof(Split)), argmap,
                         Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                         outputs[0]->machine_view.hash());
  launcher.add_region_requirement(
    RegionRequirement(inputs[0]->part, 0/*projection id*/,
      READ_ONLY, EXCLUSIVE, inputs[0]->region));
  launcher.add_field(0, FID_DATA);
  for (int i = 0; i < numOutputs; i++) {
    launcher.add_region_requirement(
      RegionRequirement(outputs[i]->part, 0/*projection id*/,
        WRITE_ONLY, EXCLUSIVE, outputs[i]->region));
    launcher.add_field(i+1, FID_DATA);
  }
  runtime->execute_index_space(ctx, launcher);
}

OpMeta* Split::init_task(const Task* task,
                         const std::vector<PhysicalRegion>& regions,
                         Context ctx, Runtime* runtime)
{
  return NULL;
}

void Split::forward(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  IndexLauncher launcher(SPLIT_FWD_TASK_ID, parallel_is,
                         TaskArgument(this, sizeof(Split)), argmap,
                         Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                         outputs[0]->machine_view.hash());
  launcher.add_region_requirement(
    RegionRequirement(inputs[0]->part, 0/*projection id*/,
      READ_ONLY, EXCLUSIVE, inputs[0]->region));
  launcher.add_field(0, FID_DATA);
  for (int i = 0; i < numOutputs; i++) {
    launcher.add_region_requirement(
      RegionRequirement(outputs[i]->part, 0/*projection id*/,
        WRITE_ONLY, EXCLUSIVE, outputs[i]->region));
    launcher.add_field(i+1, FID_DATA);
  }
  runtime->execute_index_space(ctx, launcher);
}

void calc_block_size(coord_t& num_blks,
                     coord_t& blk_size,
                     const Domain& domain,
                     int axis)
{
  num_blks = 1;
  blk_size = 1;
  for (int d = 0; d < domain.get_dim(); d++) {
    if (d <= axis)
      blk_size *= (domain.hi()[d] - domain.lo()[d] + 1);
    else
      num_blks *= (domain.hi()[d] - domain.lo()[d] + 1);
  }
}

void Split::forward_task(const Task *task,
                         const std::vector<PhysicalRegion>& regions,
                         Context ctx, Runtime *runtime)
{
  const Split* split = (Split*) task->args;
  assert(regions.size() == split->numOutputs + 1);
  assert(task->regions.size() == split->numOutputs + 1);
  Domain in_domain = runtime->get_index_space_domain(
    ctx, task->regions[0].region.get_index_space());
  float* out_ptr[MAX_NUM_OUTPUTS];
  size_t total_volume = 0;
  const float* in_ptr = helperGetTensorPointerRO<float>(
    regions[0], task->regions[0], FID_DATA, ctx, runtime);
  coord_t num_blks, in_blk_size, out_blk_size[MAX_NUM_OUTPUTS];
  calc_block_size(num_blks, in_blk_size, in_domain, split->axis);
  for (int i = 0; i < split->numOutputs; i++) {
    Domain out_domain = runtime->get_index_space_domain(
      ctx, task->regions[i+1].region.get_index_space());
    out_ptr[i] = helperGetTensorPointerWO<float>(
      regions[i+1], task->regions[i+1], FID_DATA, ctx, runtime);
    coord_t out_num_blks;
    calc_block_size(out_num_blks, out_blk_size[i], out_domain, split->axis);
    assert(out_num_blks == num_blks);
    for (int j = 0; j < out_domain.get_dim(); j++)
      if (j != split->axis) {
        assert(out_domain.hi()[j] == in_domain.hi()[j]);
        assert(out_domain.lo()[j] == in_domain.lo()[j]);
      }
    total_volume += out_domain.get_volume();
  }
  assert(total_volume == in_domain.get_volume());

  Split::forward_kernel_wrapper(out_ptr, in_ptr, out_blk_size, in_blk_size, num_blks, split->numOutputs);
}

void Split::backward(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  IndexLauncher launcher(SPLIT_BWD_TASK_ID, parallel_is,
                         TaskArgument(this, sizeof(Split)), argmap,
                         Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                         outputs[0]->machine_view.hash());
  launcher.add_region_requirement(
    RegionRequirement(inputs[0]->part_grad, 0/*projection id*/,
      READ_WRITE, EXCLUSIVE, inputs[0]->region_grad));
  launcher.add_field(0, FID_DATA);
  for (int i = 0; i < numOutputs; i++) {
    launcher.add_region_requirement(
      RegionRequirement(outputs[i]->part_grad, 0/*projection id*/,
        READ_ONLY, EXCLUSIVE, outputs[i]->region_grad));
    launcher.add_field(i+1, FID_DATA);
  }
  runtime->execute_index_space(ctx, launcher);
}

void Split::backward_task(const Task *task,
                          const std::vector<PhysicalRegion>& regions,
                          Context ctx, Runtime *runtime)
{
  const Split* split = (Split*) task->args;
  assert(regions.size() == split->numOutputs + 1);
  assert(task->regions.size() == split->numOutputs + 1);
  Domain in_grad_domain = runtime->get_index_space_domain(
    ctx, task->regions[0].region.get_index_space());
  const float* out_grad_ptr[MAX_NUM_OUTPUTS];
  size_t total_volume = 0;
  float* in_grad_ptr = helperGetTensorPointerRW<float>(
    regions[0], task->regions[0], FID_DATA, ctx, runtime);
  coord_t num_blks, in_blk_size, out_blk_size[MAX_NUM_OUTPUTS];
  calc_block_size(num_blks, in_blk_size, in_grad_domain, split->axis);
  for (int i = 0; i < split->numOutputs; i++) {
    Domain out_grad_domain = runtime->get_index_space_domain(
      ctx, task->regions[i+1].region.get_index_space());
    out_grad_ptr[i] = helperGetTensorPointerRO<float>(
      regions[i+1], task->regions[i+1], FID_DATA, ctx, runtime);
    coord_t out_num_blks;
    calc_block_size(out_num_blks, out_blk_size[i], out_grad_domain, split->axis);
    assert(out_num_blks == num_blks);
    for (int j = 0; j < out_grad_domain.get_dim(); j++)
      if (j != split->axis) {
        assert(out_grad_domain.hi()[j] == in_grad_domain.hi()[j]);
        assert(out_grad_domain.lo()[j] == in_grad_domain.lo()[j]);
      }
    total_volume += out_grad_domain.get_volume();
  }
  assert(total_volume == in_grad_domain.get_volume());

  Split::backward_kernel_wrapper(in_grad_ptr, out_grad_ptr, out_blk_size, in_blk_size, num_blks, split->numOutputs);
}

bool Split::measure_operator_cost(Simulator* sim,
                                  const ParallelConfig& pc,
                                  CostMetrics& cost_metrics) const
{
  //TODO: implement measure_forward
  ParallelTensorBase sub_output[MAX_NUM_OUTPUTS], sub_input;
  for (int i = 0; i < numOutputs; i++)
    if (!outputs[i]->get_output_sub_tensor(pc, sub_output[i], OP_SPLIT))
      return false;
  if (!inputs[0]->get_input_sub_tensor(pc, sub_input, OP_SPLIT))
    return false;
  Domain in_domain = sub_input.get_domain();
  sim->free_all();
  float* output_ptr[MAX_NUM_OUTPUTS];
  size_t total_volume = 0;
  float* input_ptr = (float*)sim->allocate(sub_input.get_volume(), DT_FLOAT);
  coord_t num_blks, in_blk_size, out_blk_size[MAX_NUM_OUTPUTS];
  calc_block_size(num_blks, in_blk_size, in_domain, axis);
  for (int i = 0; i < numOutputs; i++) {
    Domain out_domain = sub_output[i].get_domain();
    output_ptr[i] = (float*)sim->allocate(sub_output[i].get_volume(), DT_FLOAT);
    coord_t out_num_blks;
    calc_block_size(out_num_blks, out_blk_size[i], out_domain, axis);
    assert(out_num_blks == num_blks);
    for (int j = 0; j < out_domain.get_dim(); j++)
      if (j != axis) {
        assert(out_domain.hi()[j] == in_domain.hi()[j]);
        assert(out_domain.lo()[j] == in_domain.lo()[j]);
      }
    total_volume += out_domain.get_volume();
  }
  assert(total_volume == in_domain.get_volume());

  std::function<void()> forward, backward;
  forward = [&] {
    forward_kernel_wrapper(output_ptr, input_ptr, out_blk_size, in_blk_size,
                           num_blks, numOutputs);
  };
  // Assume backward has the same cost as forward
  backward = forward;

  inner_measure_operator_cost(sim, forward, backward, cost_metrics);
  if (sim->computationMode == COMP_MODE_TRAINING) {
    printf("[Measure Split] name(%s) num_elements(%zu) forward_time(%.4lf) backward_time(%.4lf)\n",
           name, sub_input.get_volume(),
           cost_metrics.forward_time,
           cost_metrics.backward_time);
  } else {
    printf("[Measure Split] name(%s) num_elements(%zu) forward_time(%.4lf)\n",
           name, sub_input.get_volume(),
           cost_metrics.forward_time);
  }
  return true;
}

}; // namespace FlexFlow