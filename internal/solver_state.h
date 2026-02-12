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

#include "pdhcg_types.h"
#include "pdhcg.h"
#include "internal_types.h"
#include "preconditioner.h"
#include "solver.h"
#include "utils.h"

#ifdef __cplusplus
extern "C"
{
#endif
    void initialize_step_size_and_primal_weight(pdhg_solver_state_t *state, const pdhg_parameters_t *params);
    
    pdhg_solver_state_t *initialize_solver_state(const pdhg_parameters_t *params,
                                                 const qp_problem_t *original_problem,
                                                 const rescale_info_t *rescale_info);
    void pdhg_solver_state_free(pdhg_solver_state_t *state);
    void rescale_info_free(rescale_info_t *info);
    void update_obj_product(pdhg_solver_state_t *state, double *primal_solution);
    double compute_xQx(pdhg_solver_state_t *state, double *primal_sol, double* primal_obj_product);
#ifdef __cplusplus
}
#endif