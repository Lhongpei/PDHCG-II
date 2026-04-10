#include "distributed_interface.h"
#include "distributed_types.h"
#include "distributed_utils.h"
#include <mpi.h>
#include <nccl.h>

void pdhcg_all_reduce_array(
    grid_context_t *ctx, double *buf, int count, pdhcg_reduce_op_t op, pdhcg_comm_scope_t scope, void *stream)
{
    if (!ctx || count <= 0)
        return;

    ncclRedOp_t nccl_op = (op == PDHCG_OP_MAX) ? ncclMax : ncclSum;
    ncclComm_t comm;

    if (scope == PDHCG_SCOPE_GLOBAL && (ctx->dims[0] * ctx->dims[1] > 1))
        comm = ctx->nccl_global;
    else if (scope == PDHCG_SCOPE_COL && ctx->dims[0] > 1)
        comm = ctx->nccl_col;
    else if (scope == PDHCG_SCOPE_ROW && ctx->dims[1] > 1)
        comm = ctx->nccl_row;
    else
        return;

    NCCL_CHECK(ncclAllReduce(buf, buf, count, ncclDouble, nccl_op, comm, (cudaStream_t)stream));
}

void pdhcg_all_reduce_scalar(
    grid_context_t *ctx, double *value, pdhcg_reduce_op_t op, pdhcg_comm_scope_t scope, bool on_device)
{
    if (!ctx)
        return;

    if (on_device)
    {
        ncclRedOp_t nccl_op = (op == PDHCG_OP_MAX) ? ncclMax : ncclSum;
        ncclComm_t comm;

        if (scope == PDHCG_SCOPE_GLOBAL && (ctx->dims[0] * ctx->dims[1] > 1))
            comm = ctx->nccl_global;
        else if (scope == PDHCG_SCOPE_COL && ctx->dims[0] > 1)
            comm = ctx->nccl_col;
        else if (scope == PDHCG_SCOPE_ROW && ctx->dims[1] > 1)
            comm = ctx->nccl_row;
        else
            return;

        NCCL_CHECK(ncclAllReduce(value, value, 1, ncclDouble, nccl_op, comm, 0));
    }
    else
    {
        MPI_Op mpi_op = (op == PDHCG_OP_MAX) ? MPI_MAX : MPI_SUM;
        MPI_Comm comm;

        if (scope == PDHCG_SCOPE_GLOBAL && (ctx->dims[0] * ctx->dims[1] > 1))
            comm = ctx->comm_global;
        else if (scope == PDHCG_SCOPE_COL && ctx->dims[0] > 1)
            comm = ctx->comm_col;
        else if (scope == PDHCG_SCOPE_ROW && ctx->dims[1] > 1)
            comm = ctx->comm_row;
        else
            return;

        double local_val = *value;
        MPI_Allreduce(&local_val, value, 1, MPI_DOUBLE, mpi_op, comm);
    }
}

int pdhcg_get_grid_p_col(struct grid_context_s *ctx)
{
    return ctx ? ctx->dims[1] : 1;
}

int pdhcg_get_grid_row_coord(struct grid_context_s *ctx)
{
    return ctx ? ctx->coords[0] : 0;
}
