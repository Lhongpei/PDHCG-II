#pragma once

#include "pdhcg_types.h"
#include "pdhcg.h"
#include "internal_types.h"
#include "preconditioner.h"
#include "presolve.h"
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