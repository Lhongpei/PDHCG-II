#pragma once

#include "cupdlpx_types.h"
#include "cupdlpx.h"
#include "internal_types.h"
#include "preconditioner.h"
#include "presolve.h"
#include "solver.h"
#include "utils.h"

#ifdef __cplusplus
extern "C"
{
#endif
    __global__ void compute_lp_next_pdhg_primal_solution_kernel(
        const double *current_primal, double *reflected_primal,
        const double *dual_product, const double *objective, const double *var_lb,
        const double *var_ub, int n, double step_size);
    __global__ void compute_lp_next_pdhg_primal_solution_major_kernel(
        const double *current_primal, double *pdhg_primal, double *reflected_primal,
        const double *dual_product, const double *objective, const double *var_lb,
        const double *var_ub, int n, double step_size, double *dual_slack);
    __global__ void compute_next_pdhg_dual_solution_kernel(
        const double *current_dual, double *reflected_dual,
        const double *primal_product, const double *const_lb,
        const double *const_ub, int n, double step_size);
    __global__ void compute_next_pdhg_dual_solution_major_kernel(
        const double *current_dual, double *pdhg_dual, double *reflected_dual,
        const double *primal_product, const double *const_lb,
        const double *const_ub, int n, double step_size);
    __global__ void compute_delta_solution_kernel(
        const double *initial_primal, const double *pdhg_primal,
        double *delta_primal, const double *initial_dual, const double *pdhg_dual,
        double *delta_dual, int n_vars, int n_cons);
    __global__ void compute_and_rescale_reduced_cost_kernel(
        double *reduced_cost,
        const double *objective,
        const double *dual_product,
        const double *variable_rescaling,
        const double objective_vector_rescaling,
        const double constraint_bound_rescaling,
        int n_vars);
    void compute_next_pdhg_primal_solution(pdhg_solver_state_t *state);
    void compute_next_pdhg_dual_solution(pdhg_solver_state_t *state);
    void halpern_update(pdhg_solver_state_t *state, double reflection_coefficient);

    void rescale_solution(pdhg_solver_state_t *state);
    
    cupdlpx_result_t *create_result_from_state(pdhg_solver_state_t *state, const lp_problem_t *original_problem);
    
    void perform_restart(pdhg_solver_state_t *state, const pdhg_parameters_t *params);
    
    void initialize_step_size_and_primal_weight(pdhg_solver_state_t *state, const pdhg_parameters_t *params);
    
    pdhg_solver_state_t *initialize_solver_state(const pdhg_parameters_t *params,
                                                 const lp_problem_t *original_problem,
                                                 const rescale_info_t *rescale_info);
    
    void compute_fixed_point_error(pdhg_solver_state_t *state);

    void perform_primal_restart(pdhg_solver_state_t *state);
    void perform_dual_restart(pdhg_solver_state_t *state);

    void primal_feasibility_polish(const pdhg_parameters_t *params, pdhg_solver_state_t *state, const pdhg_solver_state_t *ori_state);
    void dual_feasibility_polish(const pdhg_parameters_t *params, pdhg_solver_state_t *state, const pdhg_solver_state_t *ori_state);
    
    void primal_feas_polish_state_free(pdhg_solver_state_t *state);
    void dual_feas_polish_state_free(pdhg_solver_state_t *state);
    
    void feasibility_polish(const pdhg_parameters_t *params, pdhg_solver_state_t *state);
    
    void compute_primal_fixed_point_error(pdhg_solver_state_t *state);
    void compute_dual_fixed_point_error(pdhg_solver_state_t *state);
    
    pdhg_solver_state_t *initialize_primal_feas_polish_state(const pdhg_solver_state_t *original_state);
    pdhg_solver_state_t *initialize_dual_feas_polish_state(const pdhg_solver_state_t *original_state);

#ifdef __cplusplus
}
#endif