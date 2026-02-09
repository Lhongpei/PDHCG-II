#include "presolve.h"
#include "PSLP_sol.h"
#include "pdhcg.h"
#include "utils.h"
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef PSLP_VERSION
#define PSLP_VERSION "unknown"
#endif

const char *get_presolve_status_str(enum PresolveStatus_ status)
{
    switch (status)
    {
    case UNCHANGED:
        return "UNCHANGED";
    case REDUCED:
        return "REDUCED";
    case INFEASIBLE:
        return "INFEASIBLE";
    case UNBNDORINFEAS:
        return "INFEASIBLE_OR_UNBOUNDED";
    default:
        return "UNKNOWN_STATUS";
    }
}

qp_problem_t *convert_pslp_to_pdhcg(PresolvedProblem *reduced_prob)
{

    qp_problem_t *pdhcg_prob = (qp_problem_t *)safe_malloc(sizeof(qp_problem_t));
    // TODO: handle warmstart here
    pdhcg_prob->primal_start = NULL;
    pdhcg_prob->dual_start = NULL;

    pdhcg_prob->objective_constant = reduced_prob->obj_offset;
    pdhcg_prob->objective_vector = reduced_prob->c;

    pdhcg_prob->constraint_lower_bound = reduced_prob->lhs;
    pdhcg_prob->constraint_upper_bound = reduced_prob->rhs;
    pdhcg_prob->variable_lower_bound = reduced_prob->lbs;
    pdhcg_prob->variable_upper_bound = reduced_prob->ubs;

    pdhcg_prob->constraint_matrix_num_nonzeros = reduced_prob->nnz;
    pdhcg_prob->constraint_matrix = (CsrComponent *)safe_malloc(sizeof(CsrComponent));
    pdhcg_prob->constraint_matrix->row_ptr = reduced_prob->Ap;
    pdhcg_prob->constraint_matrix->col_ind = reduced_prob->Ai;
    pdhcg_prob->constraint_matrix->val = reduced_prob->Ax;

    pdhcg_prob->objective_matrix = NULL;
    pdhcg_prob->objective_matrix_num_nonzeros = 0;

    pdhcg_prob->num_variables = reduced_prob->n;
    pdhcg_prob->num_constraints = reduced_prob->m;

    return pdhcg_prob;
}

pdhcg_presolve_info_t *pslp_presolve(const qp_problem_t *original_prob, const pdhg_parameters_t *params)
{
    if (original_prob->primal_start || original_prob->dual_start)
    {
        printf("Warning: Warm-starting is currently not supported when presolve is enabled.\n"
               "The provided initial solutions will be ignored.\n");
    }
    if (params->verbose)
    {
        printf("\nRunning presolver (PSLP %s)...\n", PSLP_VERSION);
    }
    clock_t start_time = clock();

    pdhcg_presolve_info_t *info = (pdhcg_presolve_info_t *)safe_calloc(1, sizeof(pdhcg_presolve_info_t));

    // 1. Init Settings
    info->settings = default_settings();
    info->settings->verbose = false;

    // 2. Init Presolver
    info->presolver = new_presolver(
        original_prob->constraint_matrix->val,
        original_prob->constraint_matrix->col_ind,
        original_prob->constraint_matrix->row_ptr,
        original_prob->num_constraints,
        original_prob->num_variables,
        original_prob->constraint_matrix_num_nonzeros,
        original_prob->constraint_lower_bound,
        original_prob->constraint_upper_bound,
        original_prob->variable_lower_bound,
        original_prob->variable_upper_bound,
        original_prob->objective_vector,
        info->settings);

    // 3. Run Presolve
    PresolveStatus status = run_presolver(info->presolver);
    info->presolve_time = (double)(clock() - start_time) / CLOCKS_PER_SEC;

    info->presolve_status = status;
    if (params->verbose)
    {
        printf("  %-15s : %s\n", "status", get_presolve_status_str(status));
        printf("  %-15s : %.3g sec\n", "presolve time", info->presolve_time);
    }

    if (status & INFEASIBLE || status & UNBNDORINFEAS)
    {
        info->problem_solved_during_presolve = true;
        info->reduced_problem = NULL;
    }
    else
    {
        info->problem_solved_during_presolve = false;
        if (params->verbose)
        {
            printf("  %-15s : %d rows, %d columns, %d nonzeros\n",
                   "reduced problem",
                   info->presolver->reduced_prob->m,
                   info->presolver->reduced_prob->n,
                   info->presolver->reduced_prob->nnz);
        }
        info->reduced_problem = convert_pslp_to_pdhcg(info->presolver->reduced_prob);
    }
    return info;
}

pdhcg_result_t *create_result_from_presolve(const pdhcg_presolve_info_t *info, const qp_problem_t *original_prob)
{

    pdhcg_result_t *result = (pdhcg_result_t *)safe_calloc(1, sizeof(pdhcg_result_t));

    if (info->presolve_status == INFEASIBLE)
    {
        result->termination_reason = TERMINATION_REASON_PRIMAL_INFEASIBLE;
    }
    else if (info->presolve_status == UNBNDORINFEAS)
    {
        result->termination_reason = TERMINATION_REASON_INFEASIBLE_OR_UNBOUNDED;
    }
    else
    {
        result->termination_reason = TERMINATION_REASON_UNSPECIFIED;
    }
    result->num_variables = original_prob->num_variables;
    result->num_constraints = original_prob->num_constraints;
    result->num_nonzeros = original_prob->constraint_matrix_num_nonzeros;
    result->num_reduced_variables = info->presolver->reduced_prob->n;
    result->num_reduced_constraints = info->presolver->reduced_prob->m;
    result->num_reduced_nonzeros = info->presolver->reduced_prob->nnz;
    result->presolve_status = info->presolve_status;
    result->presolve_time = info->presolve_time;
    // result->presolve_stats = *(info->presolver->stats);
    // TODO: Verify if setting solution pointers to NULL affects Python/Julia bindings.
    if (result->num_variables > 0)
    {
        result->primal_solution = (double *)safe_calloc(result->num_variables, sizeof(double));
        result->reduced_cost = (double *)safe_calloc(result->num_variables, sizeof(double));
    }
    if (result->num_constraints > 0)
    {
        result->dual_solution = (double *)safe_calloc(result->num_constraints, sizeof(double));
    }
    return result;
}

void pslp_postsolve(pdhcg_presolve_info_t *info,
                    pdhcg_result_t *result,
                    const qp_problem_t *original_prob)
{
    postsolve(info->presolver,
              result->primal_solution,
              result->dual_solution,
              result->reduced_cost,
              result->primal_objective_value);

    result->num_reduced_variables = info->presolver->reduced_prob->n;
    result->num_reduced_constraints = info->presolver->reduced_prob->m;
    result->num_reduced_nonzeros = info->presolver->reduced_prob->nnz;
    result->presolve_status = info->presolve_status;

    result->primal_solution = (double *)safe_malloc(original_prob->num_variables * sizeof(double));
    result->dual_solution = (double *)safe_malloc(original_prob->num_constraints * sizeof(double));
    result->reduced_cost = (double *)safe_malloc(original_prob->num_variables * sizeof(double));

    memcpy(result->primal_solution, info->presolver->sol->x, original_prob->num_variables * sizeof(double));
    memcpy(result->dual_solution, info->presolver->sol->y, original_prob->num_constraints * sizeof(double));
    memcpy(result->reduced_cost, info->presolver->sol->z, original_prob->num_variables * sizeof(double));
    // result->primal_objective_value = info->presolver->sol->obj; // This is a bug in PSLP. We don't need to updated primal_objective_value since offset has been updated during presolve. Therefore, the original problem and reduced problem have the same objective value.
    result->presolve_time = info->presolve_time;
    // if (info->presolver->stats != NULL) {
    //     result->presolve_stats = *(info->presolver->stats);
    // }
}

void pdhcg_presolve_info_free(pdhcg_presolve_info_t *info)
{
    if (!info)
        return;
    // if (info->reduced_problem) qp_problem_free(info->reduced_problem);
    if (info->presolver)
        free_presolver(info->presolver);
    if (info->settings)
        free_settings(info->settings);
    free(info);
}