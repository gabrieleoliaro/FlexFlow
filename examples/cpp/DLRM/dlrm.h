/* Copyright 2019 Stanford
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

#include "flexflow/model.h"
#define MAX_NUM_SAMPLES 64
#define MAX_NUM_EMB 1000
#define MAX_NUM_MLPS 100
#define MAX_DATASET_PATH_LEN 1023

using namespace Legion;
using namespace FlexFlow;

struct DLRMConfig {
  DLRMConfig(void);
  int sparse_feature_size, sigmoid_bot, sigmoid_top, embedding_bag_size;
  float loss_threshold;
  std::vector<int> embedding_size, mlp_bot, mlp_top;
  std::string arch_interaction_op, dataset_path;
  int data_size;
};

struct ArgsConfig {
  int sparse_feature_size, sigmoid_bot, sigmoid_top, embedding_bag_size;
  int embedding_size, mlp_bot[MAX_NUM_MLPS], mlp_top[MAX_NUM_MLPS];
  char dataset_path[MAX_DATASET_PATH_LEN];
};

class DataLoader {
public:
  DataLoader(FFModel &ff,
             DLRMConfig const &dlrm,
             std::vector<Tensor> const &_sparse_inputs,
             Tensor _dense_input,
             Tensor _label);

  void next_batch(FFModel &ff);
  void next_label_ubatch(FlexFlow::FFModel &);
  void next_sparse_input_ubatch(FlexFlow::FFModel &, int idx);
  void next_dense_input_ubatch(FlexFlow::FFModel &);
  void shuffle();
  void reset();
  void reset_idx(void);
  static void load_entire_dataset(Task const *task,
                                  std::vector<PhysicalRegion> const &regions,
                                  Context ctx,
                                  Runtime *runtime);
  static void load_sparse_input(Task const *task,
                                std::vector<PhysicalRegion> const &regions,
                                Context ctx,
                                Runtime *runtime);
  static void load_sparse_input_cpu(Task const *task,
                                    std::vector<PhysicalRegion> const &regions,
                                    Context ctx,
                                    Runtime *runtime);
  static void load_dense_input(Task const *task,
                               std::vector<PhysicalRegion> const &regions,
                               Context ctx,
                               Runtime *runtime);
  static void load_label(Task const *task,
                         std::vector<PhysicalRegion> const &regions,
                         Context ctx,
                         Runtime *runtime);

public:
  int num_samples, next_index, next_label_index, next_dense_input_index;
  int next_sparse_input_index[MAX_NUM_EMB];
  int sparse_input_idx[MAX_NUM_EMB];
  int dense_input_idx = 0;
  int label_idx = 0;
  std::vector<Tensor> batch_sparse_inputs;
  Tensor full_sparse_input, full_dense_input, batch_dense_input, full_label,
      batch_label;
};

struct SampleIdxs {
  int num_samples;
  int idxs[MAX_NUM_SAMPLES];
};
