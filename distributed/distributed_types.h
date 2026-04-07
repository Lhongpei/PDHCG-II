/*
Copyright 2026 Hongpei Li

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#pragma once

#include <mpi.h>
#include <nccl.h>

struct grid_context_s
{
    MPI_Comm comm_global;
    MPI_Comm comm_row;
    MPI_Comm comm_col;
    ncclComm_t nccl_row;
    ncclComm_t nccl_col;
    int rank_global;
    int coords[2];
    int dims[2];
    int global_num_variables;
    int n_start;
};

typedef struct
{
    MPI_Request *reqs;
    int num_reqs;
} big_request_t;
