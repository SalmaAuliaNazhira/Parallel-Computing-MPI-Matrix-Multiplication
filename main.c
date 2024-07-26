/*
 * Code template for implementing ...
 *   matrix multiplication using Cannon's Algorithm in MPI
 *
 * The program takes three command-line arguments: fileA, fileB, and
 * fileC. The first two files contain matrix A and B as the input. The
 * third file is used to store the result matrix C as the output. The
 * program compute: C = A x B. The program assumes the matrices A, B,
 * and C are n x n matrices, the number of processors p is square, and
 * n is evenly divisible by sqrt(p).
 *
 * The files containing the matrices are all binary files and have the
 * following format. The matrix is stored in row-wise order and
 * preceded with two integers that specify the dimensions of the
 * matrix. The matrix elements are double floating point numbers.
 *
 */

#include <math.h>
#include <mpi.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <unistd.h>

/* set this parameter to reflect the cache line size of the particular
   machine you're running this program */
#define CACHE_SIZE 1024

/* in case later we decide to use another data type */
#define mpitype MPI_DOUBLE
typedef double datatype;

void sum(int n, datatype *const *c, datatype *const *partial_c_matrix);
void check_input_files(char *const *argv);
datatype **init_partial_c_matrix(int n);
datatype **init_local_c(int n, datatype **c, datatype *sc);
int cfileexists(const char *filename);
#define BLOCK_LOW(id, p, n)  ((id)*(n)/(p))

/* block decomposition macros */
void reconstruct_matrix(int ma, int na, datatype *const *a, const datatype *sa);
#define BLOCK_HIGH(id, p, n) (BLOCK_LOW((id)+1,p,n)-1)
#define BLOCK_SIZE(id, p, n) (BLOCK_HIGH(id,p,n)-BLOCK_LOW(id,p,n)+1)
#define BLOCK_OWNER(j, p, n) (((p)*((j)+1)-1)/(n))

/* print out error message and exit the program */
void my_abort(const char *fmt, ...) {
    int id;     /* process rank */
    va_list ap; /* argument list */

    va_start(ap, fmt);

    /* only process 0 reports */
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    if (!id) vprintf(fmt, ap);

    va_end(ap);

    /* all MPI processes exit at this point */
    exit(1);
}

/* return the data size in bytes */
int get_size(MPI_Datatype t) {
    if (t == MPI_BYTE) return sizeof(char);
    else if (t == MPI_DOUBLE) return sizeof(double);
    else if (t == MPI_FLOAT) return sizeof(float);
    else if (t == MPI_INT) return sizeof(int);
    else {
        printf("Error: Unrecognized argument to 'get_size'\n");
        fflush(stdout);
        MPI_Abort(MPI_COMM_WORLD, -3);
    }
    return 0;
}


/* allocate memory from heap */
void *my_malloc(int id, int bytes) {
    void *buffer;
    if ((buffer = malloc((size_t) bytes)) == NULL) {
        printf("Error: Malloc failed for process %d\n", id);
        fflush(stdout);
        MPI_Abort(MPI_COMM_WORLD, -2);
    }
    return buffer;
}

/* Read a matrix from a file. */
void read_checkerboard_matrix(
        char *s,              /* IN - File name */
        void ***subs,         /* OUT - 2D array */
        void **storage,       /* OUT - Array elements */
        MPI_Datatype dtype,   /* IN - Element type */
        int *rows,            /* OUT - Array rows */
        int *cols,            /* OUT - Array cols */
        MPI_Comm grid_comm)   /* IN - Communicator */
{
    void *buffer;         /* File buffer */
    int coords[2];      /* Coords of proc receiving
                                 next row of matrix */
    int datum_size;     /* Bytes per elements */
    int dest_id;        /* Rank of receiving proc */
    int grid_coord[2];  /* Process coords */
    int grid_id;        /* Process rank */
    int grid_period[2]; /* Wraparound */
    int grid_size[2];   /* Dimensions of grid */
    int i, j, k;
    FILE *infileptr;      /* Input file pointer */
    void *laddr;          /* Used when proc 0 gets row */
    int local_cols;     /* Matrix cols on this proc */
    int local_rows;     /* Matrix rows on this proc */
    void **lptr;           /* Pointer into 'subs' */
    int p;              /* Number of processes */
    void *raddr;          /* Address of first element
                                 to send */
    void *rptr;           /* Pointer into 'storage' */
    MPI_Status status;         /* Results of read */

    MPI_Comm_rank(grid_comm, &grid_id);
    MPI_Comm_size(grid_comm, &p);
    datum_size = get_size(dtype);

    /* Process 0 opens file, gets number of rows and
       number of cols, and broadcasts this information
       to the other processes. */

    if (grid_id == 0) {
        infileptr = fopen(s, "r");
        if (infileptr == NULL) *rows = 0;
        else {
            fread(rows, sizeof(int), 1, infileptr);
            fread(cols, sizeof(int), 1, infileptr);
        }
    }
    MPI_Bcast(rows, 1, MPI_INT, 0, grid_comm);

    if (!(*rows)) MPI_Abort(MPI_COMM_WORLD, -1);

    MPI_Bcast(cols, 1, MPI_INT, 0, grid_comm);

    /* Each process determines the size of the submatrix
       it is responsible for. */

    MPI_Cart_get(grid_comm, 2, grid_size, grid_period,
                 grid_coord);
    local_rows = BLOCK_SIZE(grid_coord[0], grid_size[0], *rows);
    local_cols = BLOCK_SIZE(grid_coord[1], grid_size[1], *cols);

    /* Dynamically allocate two-dimensional matrix 'subs' */

    *storage = my_malloc(grid_id,
                         local_rows * local_cols * datum_size);
    *subs = (void **) my_malloc(grid_id, local_rows * sizeof(void *));
    lptr = (void *) *subs;
    rptr = (void *) *storage;
    for (i = 0; i < local_rows; i++) {
        *(lptr++) = (void *) rptr;
        rptr += local_cols * datum_size;
    }

    /* Grid process 0 reads in the matrix one row at a time
       and distributes each row among the MPI processes. */

    if (grid_id == 0)
        buffer = my_malloc(grid_id, *cols * datum_size);

    /* For each row of processes in the process grid... */
    for (i = 0; i < grid_size[0]; i++) {
        coords[0] = i;

        /* For each matrix row controlled by this proc row...*/
        for (j = 0; j < BLOCK_SIZE(i, grid_size[0], *rows); j++) {

            /* Read in a row of the matrix */

            if (grid_id == 0) {
                fread(buffer, datum_size, *cols, infileptr);
            }

            /* Distribute it among process in the grid row */

            for (k = 0; k < grid_size[1]; k++) {
                coords[1] = k;

                /* Find address of first element to send */
                raddr = buffer +
                        BLOCK_LOW(k, grid_size[1], *cols) * datum_size;

                /* Determine the grid ID of the process getting
                   the subrow */
                MPI_Cart_rank(grid_comm, coords, &dest_id);

                /* Process 0 is responsible for sending...*/
                if (grid_id == 0) {

                    /* It is sending (copying) to itself */
                    if (dest_id == 0) {
                        laddr = (*subs)[j];
                        memcpy (laddr, raddr,
                                local_cols * datum_size);

                        /* It is sending to another process */
                    } else {
                        MPI_Send(raddr,
                                 BLOCK_SIZE(k, grid_size[1], *cols), dtype,
                                 dest_id, 0, grid_comm);
                    }

                    /* Process 'dest_id' is responsible for
                       receiving... */
                } else if (grid_id == dest_id) {
                    MPI_Recv((*subs)[j], local_cols, dtype, 0,
                             0, grid_comm, &status);
                }
            }
        }
    }
    if (grid_id == 0) free(buffer);
}

/*
 * Write a matrix distributed in checkerboard fashion to a file.
 */
void write_checkerboard_matrix(
        char *s,                /* IN -File name */
        void **a,               /* IN -2D matrix */
        MPI_Datatype dtype,     /* IN -Matrix element type */
        int m,                  /* IN -Matrix rows */
        int n,                  /* IN -Matrix columns */
        MPI_Comm grid_comm)     /* IN -Communicator */
{
    void *buffer;         /* Room to hold 1 matrix row */
    int coords[2];      /* Grid coords of process
                                     sending elements */
    int datum_size;     /* Bytes per matrix element */
    int elt;            /* Element index */
    int grid_coords[2]; /* Coords of this process */
    int grid_id;        /* Process rank */
    int grid_period[2]; /* Wraparound */
    int grid_size[2];   /* Dimensions of process grid */
    int i, j, k;
    FILE *outfileptr;       /* Output file pointer */
    void *raddr;          /* Address of 1st element
                                     to send */
    int src;            /* ID of proc with subrow */
    MPI_Status status;         /* Result of receive */
    int local_rows;     /* This proc's matrix row count */

    MPI_Comm_rank(grid_comm, &grid_id);
    datum_size = get_size(dtype);

    /* Matrix element type. */

    /* Set up the Cartesian topology; get the coordinates
       of this process in the grid. */
    MPI_Cart_get(grid_comm, 2, grid_size, grid_period,
                 grid_coords);

    local_rows = BLOCK_SIZE(grid_coords[0], grid_size[0], m);

    if (grid_id == 0)
        buffer = my_malloc(grid_id, n * datum_size);

    /* For each process row...*/
    for (i = 0; i < grid_size[0]; i++) {
        coords[0] = i;

        /* For each matrix row controlled by the process row...*/
        for (j = 0; j < BLOCK_SIZE(i, grid_size[0], m); j++) {

            /* Collect the matrix row on grid process 0 and
               print it */
            if (grid_coords[0] == i)
                raddr = a[j];

            coords[1] = 0;
            for (k = 0; k < grid_size[1]; k++) {
                coords[1] = k;
                MPI_Cart_rank(grid_comm, coords, &src);

                /* Process src sends elements */
                if (grid_id == 0) {
                    if (src == 0) {
                        if (grid_coords[0] == i) {
                            memcpy(buffer, raddr, BLOCK_SIZE(k, grid_size[1], n) * datum_size);
                            raddr += BLOCK_SIZE(k, grid_size[1], n) * datum_size;
                        }
                    } else {
                        MPI_Recv(buffer, BLOCK_SIZE(k, grid_size[1], n),
                                 dtype, src, 0, grid_comm, &status);
                    }
                    outfileptr = fopen(s, "a");
                    fwrite(buffer, datum_size, BLOCK_SIZE(k, grid_size[1], n),
                           outfileptr);
                    fclose(outfileptr);
                } else if (grid_id == src) {
                    MPI_Send(raddr,
                             BLOCK_SIZE(k, grid_size[1], n), dtype, 0, 0, grid_comm);
                    raddr += BLOCK_SIZE(k, grid_size[1], n) * datum_size;
                }
            }
        }
    }
    if (grid_id == 0) free(buffer);
}

/*
 *  Get the dimensions of the submatrix stored on this
 *  process.
 */
void get_local_matrix_size(
        int n,                /* IN -Global matrix size */
        int grid_size,        /* IN -Number of processes
                                         per dimension */
        int grid_coord,       /* IN -Column or row coordinate
                                         of this process */
        int *submatrix_size)  /* OUT -Number of elements */
{
    *submatrix_size = n / grid_size;
}


/* Set up the grid of processes. */
void setup_grid(
        MPI_Comm *grid_comm,  /* OUT - Communicator */
        int *grid_size,       /* OUT - Dimensions of process
                                         grid */
        int *grid_id,         /* OUT - Process rank */
        int *grid_coord)      /* OUT - Process coordinates */
{
    int free_coords[2];    /* Grid dimensions */
    int periods[2];        /* Wraparound */

    /* Set up global grid of processes. */
    MPI_Comm_size(MPI_COMM_WORLD, &grid_size[0]);
    grid_size[0] = sqrt(grid_size[0]);
    grid_size[1] = grid_size[0];

    periods[0] = 1;
    periods[1] = 1;
    MPI_Cart_create(MPI_COMM_WORLD, 2, grid_size, periods, 1, grid_comm);
    MPI_Comm_rank(*grid_comm, grid_id);
    MPI_Cart_coords(*grid_comm, *grid_id, 2, grid_coord);

    /* Set up row communicators. */
    free_coords[0] = 0;
    free_coords[1] = 1;
}

void matmul(
        datatype **a,              /* IN - Matrix A */
        datatype **b,              /* IN - Matrix B */
        datatype ***c,             /* OUT - Matrix C */
        int ldim,                  /* IN - Leading dimension */
        MPI_Comm grid_comm)        /* IN - Communicator */
{
    int i, j, k;
    int grid_id;               /* Process rank */

    /* This assumes the matrices are square and the
       blocks are also square. */
    MPI_Comm_rank(grid_comm, &grid_id);

    for (i = 0; i < ldim; i++) {
        for (j = 0; j < ldim; j++) {
            for (k = 0; k < ldim; k++)
                (*c)[i][j] += a[i][k] * b[k][j];
        }
    }
}

void allocate_2D_matrix(datatype ***mat, int rows, int cols) {
    *mat = (datatype **) malloc(rows * sizeof(datatype *));
    (*mat)[0] = (datatype *) malloc(rows * cols * sizeof(datatype));
    for (int i = 1; i < rows; i++) {
        (*mat)[i] = (*mat)[0] + i * cols;
    }
}

void free_2D_matrix(datatype **mat) {
    free(mat[0]);
    free(mat);
}

int main(int argc, char *argv[]) {
    int grid_id;
    int grid_size[2];
    int grid_coord[2];
    int i, j;
    int ldim;                   /* Leading dimension of */
    int n;
    int nblocks;
    int size;
    MPI_Comm grid_comm;

    MPI_Init(&argc, &argv);

    if (argc != 4)
        my_abort("Usage: %s <fileA> <fileB> <fileC>\n", argv[0]);

    setup_grid(&grid_comm, grid_size, &grid_id, grid_coord);

    if (grid_id == 0) {
        check_input_files(argv);
    }

    MPI_Barrier(grid_comm);

    datatype **a, **b, **c;
    void *a_storage, *b_storage, *c_storage;

    read_checkerboard_matrix(argv[1], (void ***)&a, &a_storage, mpitype, &n, &n, grid_comm);
    read_checkerboard_matrix(argv[2], (void ***)&b, &b_storage, mpitype, &n, &n, grid_comm);

    get_local_matrix_size(n, grid_size[0], grid_coord[0], &ldim);
    allocate_2D_matrix(&c, ldim, ldim);
    memset(&c[0][0], 0, ldim * ldim * sizeof(datatype));

    for (i = 0; i < grid_size[0]; i++) {
        int k = (grid_coord[0] + i) % grid_size[0];
        int dest = (grid_coord[0] + grid_size[0] - 1) % grid_size[0];
        int src = (grid_coord[0] + 1) % grid_size[0];

        matmul(a, b, &c, ldim, grid_comm);
        MPI_Sendrecv_replace(&a[0][0], ldim * ldim, mpitype, dest, 0, src, 0, grid_comm, MPI_STATUS_IGNORE);

        dest = (grid_coord[1] + grid_size[1] - 1) % grid_size[1];
        src = (grid_coord[1] + 1) % grid_size[1];
        MPI_Sendrecv_replace(&b[0][0], ldim * ldim, mpitype, dest, 0, src, 0, grid_comm, MPI_STATUS_IGNORE);
    }

    write_checkerboard_matrix(argv[3], (void **)c, mpitype, n, n, grid_comm);

    free_2D_matrix(a);
    free_2D_matrix(b);
    free_2D_matrix(c);
    free(a_storage);
    free(b_storage);
    free(c_storage);

    MPI_Finalize();

    return 0;
}
