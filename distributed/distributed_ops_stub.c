#include "distributed_interface.h"
#include <stdbool.h>

void pdhcg_all_reduce_array(
    grid_context_t *ctx, double *buf, int count, pdhcg_reduce_op_t op, pdhcg_comm_scope_t scope, void *stream)
{
    (void)ctx;
    (void)buf;
    (void)count;
    (void)op;
    (void)scope;
    (void)stream;
}

void pdhcg_all_reduce_scalar(
    grid_context_t *ctx, double *value, pdhcg_reduce_op_t op, pdhcg_comm_scope_t scope, bool on_device)
{
    (void)ctx;
    (void)value;
    (void)op;
    (void)scope;
    (void)on_device;
}

int pdhcg_get_grid_p_col(struct grid_context_s *ctx)
{
    return 1;
}

int pdhcg_get_grid_row_coord(struct grid_context_s *ctx)
{
    return 0;
}
