/*
Copyright 2025 Haihao Lu

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

#include "pdhg_core_op.h"
#include "cupdlpx.h"
#include "solver_state.h"
#include "internal_types.h"
#include "preconditioner.h"
#include "presolve.h"
#include "solver.h"
#include "utils.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <time.h>

cupdlpx_result_t *optimize(const pdhg_parameters_t *params,
                           const lp_problem_t *original_problem)
{
    print_initial_info(params, original_problem);

    cupdlpx_presolve_info_t *presolve_info = NULL;
    lp_problem_t *working_problem = deepcopy_problem(original_problem);
    if (params->presolve && working_problem->objective_matrix_num_nonzeros == 0)
    {
        presolve_info = pslp_presolve(original_problem, params);
        if (presolve_info->problem_solved_during_presolve)
        {
            cupdlpx_result_t *result = create_result_from_presolve(presolve_info, original_problem);
            cupdlpx_presolve_info_free(presolve_info);
            pdhg_final_log(result, params);
            return result;
        }
        working_problem = presolve_info->reduced_problem;
        ensure_objective_matrix_initialized(working_problem);
    }
    
    rescale_info_t *rescale_info = rescale_problem(params, working_problem);
    pdhg_solver_state_t *state = initialize_solver_state(params, working_problem, rescale_info);

    rescale_info_free(rescale_info);
    initialize_step_size_and_primal_weight(state, params);
    clock_t start_time = clock();
    bool do_restart = false;
    while (state->total_count < params->termination_criteria.iteration_limit)
    {
        if ((state->is_this_major_iteration || state->total_count == 0) ||
            (state->total_count % get_print_frequency(state->total_count) == 0))
        {
            compute_residual(state, params->optimality_norm);
            if (state->is_this_major_iteration &&
                state->total_count < 3 * params->termination_evaluation_frequency)
            {
                compute_infeasibility_information(state);
            }

            state->cumulative_time_sec =
                (double)(clock() - start_time) / CLOCKS_PER_SEC;

            check_termination_criteria(state, &params->termination_criteria);
            display_iteration_stats(state, params->verbose);
            if (state->termination_reason != TERMINATION_REASON_UNSPECIFIED) {
                break;
            }
        }

        if ((state->is_this_major_iteration || state->total_count == 0))
        {
            do_restart =
                should_do_adaptive_restart(state, &params->restart_params,
                                           params->termination_evaluation_frequency);
            if (do_restart)
                perform_restart(state, params);
        }

        state->is_this_major_iteration =
            ((state->total_count + 1) % params->termination_evaluation_frequency) ==
            0;

        compute_next_pdhg_primal_solution(state);
        compute_next_pdhg_dual_solution(state);

        if (state->is_this_major_iteration || do_restart)
        {
            compute_fixed_point_error(state);
            if (do_restart)
            {
                state->initial_fixed_point_error = state->fixed_point_error;
                do_restart = false;
            }
        }
        halpern_update(state, params->reflection_coefficient);

        state->inner_count++;
        state->total_count++;
    }

    if (state->termination_reason == TERMINATION_REASON_UNSPECIFIED)
    {
        state->termination_reason = TERMINATION_REASON_ITERATION_LIMIT;
        compute_residual(state, params->optimality_norm);
        display_iteration_stats(state, params->verbose);
    }

    if (params->feasibility_polishing &&
        state->termination_reason != TERMINATION_REASON_DUAL_INFEASIBLE &&
        state->termination_reason != TERMINATION_REASON_PRIMAL_INFEASIBLE)
    {
        feasibility_polish(params, state);
    }

    cupdlpx_result_t *result = create_result_from_state(state, original_problem);

    if (params->presolve && presolve_info)
    {
        pslp_postsolve(presolve_info, result, original_problem);
        cupdlpx_presolve_info_free(presolve_info);
    }

    pdhg_final_log(result, params);
    pdhg_solver_state_free(state);
    return result;
}

