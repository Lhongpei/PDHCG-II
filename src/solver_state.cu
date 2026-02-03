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

static void initialize_quadratic_obj_term(pdhg_solver_state_t *state, const lp_problem_t *problem)
{
    state->quadratic_objective_term =
        (quadratic_objective_term_t *)safe_malloc(sizeof(quadratic_objective_term_t));

    if (!problem->objective_matrix || problem->objective_matrix_num_nonzeros == 0)
    {
        state->quadratic_objective_term->quad_obj_type = PDHCG_NON_Q;
    }
    else
    {
        state->quadratic_objective_term->quad_obj_type = 
            detect_q_type(problem->objective_matrix, problem->num_variables);
    }

    int n = problem->num_variables;
    if(state->quadratic_objective_term->quad_obj_type == PDHCG_NON_Q) return;

    size_t var_bytes = n * sizeof(double);
    ALLOC_ZERO(state->quadratic_objective_term->primal_obj_product, var_bytes);
    CUSPARSE_CHECK(cusparseCreateDnVec(&state->quadratic_objective_term->vec_primal_obj_prod,
        state->num_variables,
        state->quadratic_objective_term->primal_obj_product, CUDA_R_64F));

    switch (state->quadratic_objective_term->quad_obj_type)
    {
    case PDHCG_NON_Q:
        break;
    case PDHCG_DIAG_Q:
    {
        double *h_diag = (double *)safe_calloc(n, sizeof(double)); 

        if (state->quadratic_objective_term->quad_obj_type == PDHCG_DIAG_Q && problem->objective_matrix)
        {
            CsrComponent *csr = problem->objective_matrix;
            if (csr->row_ptr && csr->col_ind && csr->val) 
            {
                for (int i = 0; i < n; ++i)
                {
                    for (int k = csr->row_ptr[i]; k < csr->row_ptr[i + 1]; ++k)
                    {
                        int col = csr->col_ind[k];
                        if (col < n) {
                            h_diag[col] = csr->val[k];
                        }
                    }
                }
            }
        }

        ALLOC_AND_COPY(state->quadratic_objective_term->diagonal_objective_matrix,
                       h_diag,
                       n * sizeof(double));
        
        free(h_diag);
        state->quadratic_objective_term->objective_matrix = NULL;
        break;
    }

    case PDHCG_SPARSE_Q:
    {
        state->quadratic_objective_term->objective_matrix = 
            (cu_sparse_matrix_csr_t *)safe_malloc(sizeof(cu_sparse_matrix_csr_t));
        
        memset(state->quadratic_objective_term->objective_matrix, 0, sizeof(cu_sparse_matrix_csr_t));
        state->quadratic_objective_term->objective_matrix->num_rows = n;
        state->quadratic_objective_term->objective_matrix->num_cols = n;
        state->quadratic_objective_term->objective_matrix->num_nonzeros = problem->objective_matrix_num_nonzeros;

        ALLOC_AND_COPY_CSR(state->quadratic_objective_term->objective_matrix, 
                           problem->objective_matrix,
                           problem->num_variables, 
                           problem->objective_matrix_num_nonzeros);

        CUSPARSE_CHECK(cusparseCreateCsr(
            &state->quadratic_objective_term->matQ, state->num_variables, state->num_variables,
            state->quadratic_objective_term->objective_matrix->num_nonzeros, state->quadratic_objective_term->objective_matrix->row_ptr,
            state->quadratic_objective_term->objective_matrix->col_ind, state->quadratic_objective_term->objective_matrix->val,
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
            CUDA_R_64F));

        size_t primal_spmv_buffer_size;
        CUSPARSE_CHECK(cusparseSpMV_bufferSize(
            state->sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &HOST_ONE,
            state->quadratic_objective_term->matQ, state->vec_primal_sol, &HOST_ZERO, state->quadratic_objective_term->vec_primal_obj_prod,
            CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG2, &primal_spmv_buffer_size));
        CUDA_CHECK(cudaMalloc(&state->quadratic_objective_term->primal_obj_spmv_buffer, primal_spmv_buffer_size));
        CUSPARSE_CHECK(cusparseSpMV_preprocess(
            state->sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &HOST_ONE,
            state->quadratic_objective_term->matQ, state->vec_primal_sol, &HOST_ZERO, state->quadratic_objective_term->vec_primal_obj_prod,
            CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG2, state->quadratic_objective_term->primal_obj_spmv_buffer));

    CUDA_CHECK(cudaGetLastError());
                           
        state->quadratic_objective_term->diagonal_objective_matrix = NULL;
        break;
    }

    default:
        fprintf(stderr, "Error: Unknown Quadratic Objective Type detected.\n");
        exit(EXIT_FAILURE);
    }
}

static void initialize_inner_solver(pdhg_solver_state_t *state)
{
    state->inner_solver = 
            (inner_solver_t *)safe_malloc(sizeof(inner_solver_t));
    if (!(state->quadratic_objective_term->quad_obj_type == PDHCG_NON_Q || state->quadratic_objective_term->quad_obj_type == PDHCG_DIAG_Q))
    {
        ALLOC_ZERO(state->inner_solver->primal_buffer, state->num_variables * sizeof(double));
        ALLOC_ZERO(state->inner_solver->dual_buffer, state->num_constraints * sizeof(double));
    }
    switch (state->quadratic_objective_term->quad_obj_type)
    {
    case PDHCG_NON_Q:
        break;
    case PDHCG_DIAG_Q:
        break;
    case PDHCG_SPARSE_Q:
        
        state->inner_solver->bb_step_size = 
            (bb_step_size_t *)safe_malloc(sizeof(bb_step_size_t));
        ALLOC_ZERO(state->inner_solver->bb_step_size->gradient, state->num_variables * sizeof(double));
        ALLOC_ZERO(state->inner_solver->bb_step_size->direction, state->num_variables * sizeof(double));
        state->inner_solver->bb_step_size->iteration_limit = 10;
        state->inner_solver->bb_step_size->tol = 1e-3;
        break;
    default:
        fprintf(stderr, "Error: Unknown Quadratic Objective Type detected.\n");
        exit(EXIT_FAILURE);
    }
}

__global__ void element_wise_mul_kernel(
    const double* __restrict__ A, 
    const double* __restrict__ B, 
    double* __restrict__ C, 
    int n) 
{
    for (int idx = blockDim.x * blockIdx.x + threadIdx.x; 
         idx < n; 
         idx += blockDim.x * gridDim.x) 
    {
        C[idx] = A[idx] * B[idx];
    }
}

void update_obj_product(pdhg_solver_state_t *state, double *primal_solution)
{
    switch (state->quadratic_objective_term->quad_obj_type)
    {
        case PDHCG_NON_Q:
            return;
        case PDHCG_SPARSE_Q:
            CUSPARSE_CHECK(cusparseDnVecSetValues(state->vec_primal_sol,
                                            primal_solution));
            CUSPARSE_CHECK(
                cusparseDnVecSetValues(state->quadratic_objective_term->vec_primal_obj_prod, 
                                        state->quadratic_objective_term->primal_obj_product));
            CUSPARSE_CHECK(cusparseSpMV(
                state->sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &HOST_ONE,
                state->quadratic_objective_term->matQ, state->vec_primal_sol, &HOST_ZERO, state->quadratic_objective_term->vec_primal_obj_prod,
                CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG2, state->quadratic_objective_term->primal_obj_spmv_buffer));
            return;
        case PDHCG_DIAG_Q:
            element_wise_mul_kernel<<<state->num_blocks_primal,
                                    THREADS_PER_BLOCK>>>(
                                        state->quadratic_objective_term->diagonal_objective_matrix, primal_solution, 
                                        state->quadratic_objective_term->primal_obj_product, state->num_variables);
            return;
        default:
            fprintf(stderr, "Error: Unknown Quadratic Objective Type detected.\n");
            exit(EXIT_FAILURE);
    }
}

double compute_xQx(pdhg_solver_state_t *state, double *primal_sol, double* primal_obj_product)
{
    if (state->quadratic_objective_term->quad_obj_type == PDHCG_NON_Q) return 0.0;
        
    double xQx = 0.0;
    CUBLAS_CHECK(cublasDdot(
        state->blas_handle, state->num_variables, primal_sol, 1,
        primal_obj_product, 1, &xQx));
    return xQx;
}

static void decide_problem_type(pdhg_solver_state_t *state)
{
    if (state->quadratic_objective_term->quad_obj_type == PDHCG_NON_Q)
        state->problem_type = LP;
    else 
        state->problem_type = CONVEX_QP;
}

pdhg_solver_state_t *
initialize_solver_state(const pdhg_parameters_t *params,
                        const lp_problem_t *working_problem,
                        const rescale_info_t *rescale_info)
{
    pdhg_solver_state_t *state =
        (pdhg_solver_state_t *)safe_calloc(1, sizeof(pdhg_solver_state_t));

    int n_vars = working_problem->num_variables;
    int n_cons = working_problem->num_constraints;
    size_t var_bytes = n_vars * sizeof(double);
    size_t con_bytes = n_cons * sizeof(double);

    state->num_variables = n_vars;
    state->num_constraints = n_cons;
    state->objective_constant = working_problem->objective_constant;

    state->constraint_matrix =
        (cu_sparse_matrix_csr_t *)safe_malloc(sizeof(cu_sparse_matrix_csr_t));
    state->constraint_matrix_t =
        (cu_sparse_matrix_csr_t *)safe_malloc(sizeof(cu_sparse_matrix_csr_t));

    state->constraint_matrix->num_rows = n_cons;
    state->constraint_matrix->num_cols = n_vars;
    state->constraint_matrix->num_nonzeros = working_problem->constraint_matrix_num_nonzeros;

    state->constraint_matrix_t->num_rows = n_vars;
    state->constraint_matrix_t->num_cols = n_cons;
    state->constraint_matrix_t->num_nonzeros = working_problem->constraint_matrix_num_nonzeros;

    state->termination_reason = TERMINATION_REASON_UNSPECIFIED;

    state->rescaling_time_sec = rescale_info->rescaling_time_sec;


    ALLOC_AND_COPY_CSR(state->constraint_matrix, 
                   rescale_info->scaled_problem->constraint_matrix,
                   rescale_info->scaled_problem->num_constraints, 
                   rescale_info->scaled_problem->constraint_matrix_num_nonzeros); 
    
    CUDA_CHECK(cudaMalloc(&state->constraint_matrix_t->row_ptr,
                          (n_vars + 1) * sizeof(int)));
    CUDA_CHECK(
        cudaMalloc(&state->constraint_matrix_t->col_ind,
                   rescale_info->scaled_problem->constraint_matrix_num_nonzeros *
                       sizeof(int)));
    CUDA_CHECK(
        cudaMalloc(&state->constraint_matrix_t->val,
                   rescale_info->scaled_problem->constraint_matrix_num_nonzeros *
                       sizeof(double)));

    CUSPARSE_CHECK(cusparseCreate(&state->sparse_handle));
    CUBLAS_CHECK(cublasCreate(&state->blas_handle));
    CUBLAS_CHECK(
        cublasSetPointerMode(state->blas_handle, CUBLAS_POINTER_MODE_HOST));

    size_t buffer_size = 0;
    void *buffer = nullptr;
    CUSPARSE_CHECK(cusparseCsr2cscEx2_bufferSize(
        state->sparse_handle, state->constraint_matrix->num_rows,
        state->constraint_matrix->num_cols,
        state->constraint_matrix->num_nonzeros, state->constraint_matrix->val,
        state->constraint_matrix->row_ptr, state->constraint_matrix->col_ind,
        state->constraint_matrix_t->val, state->constraint_matrix_t->row_ptr,
        state->constraint_matrix_t->col_ind, CUDA_R_64F, CUSPARSE_ACTION_NUMERIC,
        CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG_DEFAULT, &buffer_size));
    CUDA_CHECK(cudaMalloc(&buffer, buffer_size));

    CUSPARSE_CHECK(cusparseCsr2cscEx2(
        state->sparse_handle, state->constraint_matrix->num_rows,
        state->constraint_matrix->num_cols,
        state->constraint_matrix->num_nonzeros, state->constraint_matrix->val,
        state->constraint_matrix->row_ptr, state->constraint_matrix->col_ind,
        state->constraint_matrix_t->val, state->constraint_matrix_t->row_ptr,
        state->constraint_matrix_t->col_ind, CUDA_R_64F, CUSPARSE_ACTION_NUMERIC,
        CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG_DEFAULT, buffer));

    CUDA_CHECK(cudaFree(buffer));

    ALLOC_AND_COPY(state->variable_lower_bound,
                   rescale_info->scaled_problem->variable_lower_bound, var_bytes);
    ALLOC_AND_COPY(state->variable_upper_bound,
                   rescale_info->scaled_problem->variable_upper_bound, var_bytes);
    ALLOC_AND_COPY(state->objective_vector,
                   rescale_info->scaled_problem->objective_vector, var_bytes);
    ALLOC_AND_COPY(state->constraint_lower_bound,
                   rescale_info->scaled_problem->constraint_lower_bound,
                   con_bytes);
    ALLOC_AND_COPY(state->constraint_upper_bound,
                   rescale_info->scaled_problem->constraint_upper_bound,
                   con_bytes);
    ALLOC_AND_COPY(state->constraint_rescaling, rescale_info->con_rescale,
                   con_bytes);
    ALLOC_AND_COPY(state->variable_rescaling, rescale_info->var_rescale,
                   var_bytes);

    state->constraint_bound_rescaling = rescale_info->con_bound_rescale;
    state->objective_vector_rescaling = rescale_info->obj_vec_rescale;


    ALLOC_ZERO(state->initial_primal_solution, var_bytes);
    ALLOC_ZERO(state->current_primal_solution, var_bytes);
    ALLOC_ZERO(state->pdhg_primal_solution, var_bytes);
    ALLOC_ZERO(state->reflected_primal_solution, var_bytes);
    ALLOC_ZERO(state->dual_product, var_bytes);
    ALLOC_ZERO(state->dual_slack, var_bytes);
    ALLOC_ZERO(state->dual_residual, var_bytes);
    ALLOC_ZERO(state->delta_primal_solution, var_bytes);

    ALLOC_ZERO(state->initial_dual_solution, con_bytes);
    ALLOC_ZERO(state->current_dual_solution, con_bytes);
    ALLOC_ZERO(state->pdhg_dual_solution, con_bytes);
    ALLOC_ZERO(state->reflected_dual_solution, con_bytes);
    ALLOC_ZERO(state->primal_product, con_bytes);
    ALLOC_ZERO(state->primal_slack, con_bytes);
    ALLOC_ZERO(state->primal_residual, con_bytes);
    ALLOC_ZERO(state->delta_dual_solution, con_bytes);

    if (working_problem->primal_start)
    {
        double *rescaled = (double *)safe_malloc(var_bytes);
        for (int i = 0; i < n_vars; ++i)
            rescaled[i] = working_problem->primal_start[i] *
                          rescale_info->var_rescale[i] *
                          rescale_info->con_bound_rescale;
        CUDA_CHECK(cudaMemcpy(state->initial_primal_solution, rescaled, var_bytes,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(state->current_primal_solution, rescaled, var_bytes,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(state->pdhg_primal_solution, rescaled, var_bytes,
                              cudaMemcpyHostToDevice));
        free(rescaled);
    }
    if (working_problem->dual_start)
    {
        double *rescaled = (double *)safe_malloc(con_bytes);
        for (int i = 0; i < n_cons; ++i)
            rescaled[i] = working_problem->dual_start[i] *
                          rescale_info->con_rescale[i] *
                          rescale_info->obj_vec_rescale;
        CUDA_CHECK(cudaMemcpy(state->initial_dual_solution, rescaled, con_bytes,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(state->current_dual_solution, rescaled, con_bytes,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(state->pdhg_dual_solution, rescaled, con_bytes,
                              cudaMemcpyHostToDevice));
        free(rescaled);
    }

    double *temp_host = (double *)safe_malloc(fmax(var_bytes, con_bytes));
    for (int i = 0; i < n_cons; ++i)
        temp_host[i] =
            isfinite(rescale_info->scaled_problem->constraint_lower_bound[i])
                ? rescale_info->scaled_problem->constraint_lower_bound[i]
                : 0.0;
    ALLOC_AND_COPY(state->constraint_lower_bound_finite_val, temp_host,
                   con_bytes);
    for (int i = 0; i < n_cons; ++i)
        temp_host[i] =
            isfinite(rescale_info->scaled_problem->constraint_upper_bound[i])
                ? rescale_info->scaled_problem->constraint_upper_bound[i]
                : 0.0;
    ALLOC_AND_COPY(state->constraint_upper_bound_finite_val, temp_host,
                   con_bytes);
    for (int i = 0; i < n_vars; ++i)
        temp_host[i] =
            isfinite(rescale_info->scaled_problem->variable_lower_bound[i])
                ? rescale_info->scaled_problem->variable_lower_bound[i]
                : 0.0;
    ALLOC_AND_COPY(state->variable_lower_bound_finite_val, temp_host, var_bytes);
    for (int i = 0; i < n_vars; ++i)
        temp_host[i] =
            isfinite(rescale_info->scaled_problem->variable_upper_bound[i])
                ? rescale_info->scaled_problem->variable_upper_bound[i]
                : 0.0;
    ALLOC_AND_COPY(state->variable_upper_bound_finite_val, temp_host, var_bytes);
    free(temp_host);

    double sum_of_squares = 0.0;
    double max_val = 0.0;
    double val = 0.0;

    for (int i = 0; i < n_vars; ++i)
    {
        if (params->optimality_norm == NORM_TYPE_L_INF) {
            val = fabs(working_problem->objective_vector[i]);
            if (val > max_val) max_val = val;
        } else {
            sum_of_squares += working_problem->objective_vector[i] * working_problem->objective_vector[i];
        }
    }

    if (params->optimality_norm == NORM_TYPE_L_INF) {
        state->objective_vector_norm = max_val;
    } else {
        state->objective_vector_norm = sqrt(sum_of_squares);
    }

    sum_of_squares = 0.0;
    max_val = 0.0;
    val = 0.0;

    for (int i = 0; i < n_cons; ++i)
    {
        double lower = working_problem->constraint_lower_bound[i];
        double upper = working_problem->constraint_upper_bound[i];

        if (params->optimality_norm == NORM_TYPE_L_INF) {
            if (isfinite(lower) && (lower != upper)) {
                val = fabs(lower);
                if (val > max_val) max_val = val;
            }
            if (isfinite(upper)) {
                val = fabs(upper);
                if (val > max_val) max_val = val;
            }
        } else {
            if (isfinite(lower) && (lower != upper)) {
                sum_of_squares += lower * lower;
            }
            if (isfinite(upper)) {
                sum_of_squares += upper * upper;
            }
        }
    }

    if (params->optimality_norm == NORM_TYPE_L_INF) {
        state->constraint_bound_norm = max_val;
    } else {
        state->constraint_bound_norm = sqrt(sum_of_squares);
    }

    state->num_blocks_primal =
        (state->num_variables + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    state->num_blocks_dual =
        (state->num_constraints + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    state->num_blocks_primal_dual =
        (state->num_variables + state->num_constraints + THREADS_PER_BLOCK - 1) /
        THREADS_PER_BLOCK;

    state->best_primal_dual_residual_gap = INFINITY;
    state->last_trial_fixed_point_error = INFINITY;
    state->step_size = 0.0;
    state->is_this_major_iteration = false;

    size_t primal_spmv_buffer_size;
    size_t dual_spmv_buffer_size;

    CUSPARSE_CHECK(cusparseCreateCsr(
        &state->matA, state->num_constraints, state->num_variables,
        state->constraint_matrix->num_nonzeros, state->constraint_matrix->row_ptr,
        state->constraint_matrix->col_ind, state->constraint_matrix->val,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
        CUDA_R_64F));

    CUDA_CHECK(cudaGetLastError());

    CUSPARSE_CHECK(cusparseCreateCsr(
        &state->matAt, state->num_variables, state->num_constraints,
        state->constraint_matrix_t->num_nonzeros,
        state->constraint_matrix_t->row_ptr, state->constraint_matrix_t->col_ind,
        state->constraint_matrix_t->val, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
    CUDA_CHECK(cudaGetLastError());

    CUSPARSE_CHECK(cusparseCreateDnVec(&state->vec_primal_sol,
                                       state->num_variables,
                                       state->pdhg_primal_solution, CUDA_R_64F));
    CUSPARSE_CHECK(cusparseCreateDnVec(&state->vec_dual_sol,
                                       state->num_constraints,
                                       state->pdhg_dual_solution, CUDA_R_64F));
    CUSPARSE_CHECK(cusparseCreateDnVec(&state->vec_primal_prod,
                                       state->num_constraints,
                                       state->primal_product, CUDA_R_64F));
    CUSPARSE_CHECK(cusparseCreateDnVec(&state->vec_dual_prod,
                                       state->num_variables, state->dual_product,
                                       CUDA_R_64F));
    CUSPARSE_CHECK(cusparseSpMV_bufferSize(
        state->sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &HOST_ONE,
        state->matA, state->vec_primal_sol, &HOST_ZERO, state->vec_primal_prod,
        CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG2, &primal_spmv_buffer_size));

    CUSPARSE_CHECK(cusparseSpMV_bufferSize(
        state->sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &HOST_ONE,
        state->matAt, state->vec_dual_sol, &HOST_ZERO, state->vec_dual_prod,
        CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG2, &dual_spmv_buffer_size));
    CUDA_CHECK(cudaMalloc(&state->primal_spmv_buffer, primal_spmv_buffer_size));
    CUSPARSE_CHECK(cusparseSpMV_preprocess(
        state->sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &HOST_ONE,
        state->matA, state->vec_primal_sol, &HOST_ZERO, state->vec_primal_prod,
        CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG2, state->primal_spmv_buffer));

    CUDA_CHECK(cudaMalloc(&state->dual_spmv_buffer, dual_spmv_buffer_size));
    CUSPARSE_CHECK(cusparseSpMV_preprocess(
        state->sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &HOST_ONE,
        state->matAt, state->vec_dual_sol, &HOST_ZERO, state->vec_dual_prod,
        CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG2, state->dual_spmv_buffer));

    initialize_quadratic_obj_term(state, rescale_info->scaled_problem);
    initialize_inner_solver(state);

    CUDA_CHECK(
        cudaMalloc(&state->ones_primal_d, state->num_variables * sizeof(double)));
    CUDA_CHECK(
        cudaMalloc(&state->ones_dual_d, state->num_constraints * sizeof(double)));

    double *ones_primal_h =
        (double *)safe_malloc(state->num_variables * sizeof(double));
    for (int i = 0; i < state->num_variables; ++i)
        ones_primal_h[i] = 1.0;
    CUDA_CHECK(cudaMemcpy(state->ones_primal_d, ones_primal_h,
                          state->num_variables * sizeof(double),
                          cudaMemcpyHostToDevice));
    free(ones_primal_h);

    double *ones_dual_h =
        (double *)safe_malloc(state->num_constraints * sizeof(double));
    for (int i = 0; i < state->num_constraints; ++i)
        ones_dual_h[i] = 1.0;
    CUDA_CHECK(cudaMemcpy(state->ones_dual_d, ones_dual_h,
                          state->num_constraints * sizeof(double),
                          cudaMemcpyHostToDevice));
    decide_problem_type(state);
    free(ones_dual_h);
    if (params->verbose) {
        printf("---------------------------------------------------------------------"
            "------------------\n");
        printf("Problem Type: %s\n", problem_type_to_string(state->problem_type));
        printf("Quadratic Objective Matrix Type: %s\n", quad_obj_type_to_string(state->quadratic_objective_term->quad_obj_type));
        printf("---------------------------------------------------------------------"
            "------------------\n");
        printf("%s | %s | %s | %s \n", "   runtime    ", "    objective     ",
            "  absolute residuals   ", "  relative residuals   ");
        printf("%s %s | %s %s | %s %s %s | %s %s %s \n", "  iter", "  time ",
            " pr obj ", "  du obj ", " pr res", " du res", "  gap  ", " pr res",
            " du res", "  gap  ");
        printf("---------------------------------------------------------------------"
            "------------------\n");
    }

    return state;
}
void pdhg_solver_state_free(pdhg_solver_state_t *state)
{
    if (state == NULL)
    {
        return;
    }

    if (state->variable_lower_bound)
        CUDA_CHECK(cudaFree(state->variable_lower_bound));
    if (state->variable_upper_bound)
        CUDA_CHECK(cudaFree(state->variable_upper_bound));
    if (state->objective_vector)
        CUDA_CHECK(cudaFree(state->objective_vector));
    if (state->constraint_matrix->row_ptr)
        CUDA_CHECK(cudaFree(state->constraint_matrix->row_ptr));
    if (state->constraint_matrix->col_ind)
        CUDA_CHECK(cudaFree(state->constraint_matrix->col_ind));
    if (state->constraint_matrix->val)
        CUDA_CHECK(cudaFree(state->constraint_matrix->val));
    if (state->constraint_matrix_t->row_ptr)
        CUDA_CHECK(cudaFree(state->constraint_matrix_t->row_ptr));
    if (state->constraint_matrix_t->col_ind)
        CUDA_CHECK(cudaFree(state->constraint_matrix_t->col_ind));
    if (state->constraint_matrix_t->val)
        CUDA_CHECK(cudaFree(state->constraint_matrix_t->val));
    if (state->constraint_lower_bound)
        CUDA_CHECK(cudaFree(state->constraint_lower_bound));
    if (state->constraint_upper_bound)
        CUDA_CHECK(cudaFree(state->constraint_upper_bound));
    if (state->constraint_lower_bound_finite_val)
        CUDA_CHECK(cudaFree(state->constraint_lower_bound_finite_val));
    if (state->constraint_upper_bound_finite_val)
        CUDA_CHECK(cudaFree(state->constraint_upper_bound_finite_val));
    if (state->variable_lower_bound_finite_val)
        CUDA_CHECK(cudaFree(state->variable_lower_bound_finite_val));
    if (state->variable_upper_bound_finite_val)
        CUDA_CHECK(cudaFree(state->variable_upper_bound_finite_val));
    if (state->initial_primal_solution)
        CUDA_CHECK(cudaFree(state->initial_primal_solution));
    if (state->current_primal_solution)
        CUDA_CHECK(cudaFree(state->current_primal_solution));
    if (state->pdhg_primal_solution)
        CUDA_CHECK(cudaFree(state->pdhg_primal_solution));
    if (state->reflected_primal_solution)
        CUDA_CHECK(cudaFree(state->reflected_primal_solution));
    if (state->dual_product)
        CUDA_CHECK(cudaFree(state->dual_product));
    if (state->initial_dual_solution)
        CUDA_CHECK(cudaFree(state->initial_dual_solution));
    if (state->current_dual_solution)
        CUDA_CHECK(cudaFree(state->current_dual_solution));
    if (state->pdhg_dual_solution)
        CUDA_CHECK(cudaFree(state->pdhg_dual_solution));
    if (state->reflected_dual_solution)
        CUDA_CHECK(cudaFree(state->reflected_dual_solution));
    if (state->primal_product)
        CUDA_CHECK(cudaFree(state->primal_product));
    if (state->constraint_rescaling)
        CUDA_CHECK(cudaFree(state->constraint_rescaling));
    if (state->variable_rescaling)
        CUDA_CHECK(cudaFree(state->variable_rescaling));
    if (state->primal_slack)
        CUDA_CHECK(cudaFree(state->primal_slack));
    if (state->dual_slack)
        CUDA_CHECK(cudaFree(state->dual_slack));
    if (state->primal_residual)
        CUDA_CHECK(cudaFree(state->primal_residual));
    if (state->dual_residual)
        CUDA_CHECK(cudaFree(state->dual_residual));
    if (state->delta_primal_solution)
        CUDA_CHECK(cudaFree(state->delta_primal_solution));
    if (state->delta_dual_solution)
        CUDA_CHECK(cudaFree(state->delta_dual_solution));
    if (state->ones_primal_d)
        CUDA_CHECK(cudaFree(state->ones_primal_d));
    if (state->ones_dual_d)
        CUDA_CHECK(cudaFree(state->ones_dual_d));

    free(state);
}

void rescale_info_free(rescale_info_t *info)
{
    if (info == NULL)
    {
        return;
    }

    lp_problem_free(info->scaled_problem);
    free(info->con_rescale);
    free(info->var_rescale);

    free(info);
}
