/*
Copyright 2025 Haihao Lu
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

#include "internal_types.h"
#include "pdhcg.h"
#include "pdhcg_types.h"
#include "preconditioner.h"
#include "solver.h"
#include "utils.h"

#ifdef __cplusplus
extern "C"
{
#endif
    processed_qp_problem_t *preprocess_qp_problem(const qp_problem_t *raw_problem);
    void free_processed_qp_problem(processed_qp_problem_t *processed);
    void initialize_step_size_and_primal_weight(pdhg_solver_state_t *state, const pdhg_parameters_t *params);

    pdhg_solver_state_t *initialize_solver_state(const pdhg_parameters_t *params,
                                                 const processed_qp_problem_t *working_problem,
                                                 const rescale_info_t *rescale_info,
                                                 grid_context_t *grid_context);

    void pdhg_solver_state_free(pdhg_solver_state_t *state);
    void rescale_info_free(rescale_info_t *info);

#ifdef __cplusplus
}
#endif
