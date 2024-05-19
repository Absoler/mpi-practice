#define BUNCHSIZE 4
#define EPS 1.0E-10

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>
#include <stdbool.h>

void print_matrix(double **T, int rows, int cols);
double **copy_matrix(double **T, int rows, int cols);
void gaus_elim_seq(double **a, double *a_0, int n);
double *copy_vector(double *T, int size);
bool is_matrix_equal(double **a, double **b, int n);
void gaus_elim_loop_unroll(double **a, double *a_0, int n);
void mpi_ge(double **a, double *a_0, int n);
// void gaus_elim_block(double **a, double *a_0, int n);
double get_seconds(struct timeval start, struct timeval end);

int main(int agrc, char *agrv[])
{
    double *a0; // auxiliary 1D for 2D matrix a
    double **a; // 2D matrix for sequential computation
    int i, j;

    int n; // input size

    struct timeval start_time, end_time;
    long seconds, microseconds;
    double elapsed_seq, elapsed_loop_unroll;

    if (agrc == 2)
    {
        n = atoi(agrv[1]);
        printf("The matrix size:  %d * %d \n", n, n);
    }
    else
    {
        printf("Usage: %s n\n\n"
               " n: the matrix size\n\n",
               agrv[0]);
        return 1;
    }

    printf("Creating and initializing matrices...\n\n");
    /*** Allocate contiguous memory for 2D matrices ***/
    a0 = (double *)malloc(n * n * sizeof(double));
    a = (double **)malloc(n * sizeof(double *));
    for (i = 0; i < n; i++)
    {
        a[i] = a0 + i * n;
    }

    srand(time(0));
    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++)
            a[i][j] = (double)rand() / RAND_MAX;

    
    a[0][0] = 2;
    a[0][1] = 1;
    a[0][2] = 2;
    a[1][0] = 0.5;
    a[1][1] = 1.5;
    a[1][2] = 2;
    a[2][0] = 2;
    a[2][1] = -0.67;
    a[2][2] = -0.67;


    printf("matrix a: \n");
    print_matrix(a, n, n);

    double **a_sequential = copy_matrix(a, n, n);
    double *a0_sequential = copy_vector(a0, n * n);
    printf("Starting sequential computation...\n\n");
    /**** Sequential computation *****/
    gettimeofday(&start_time, 0);
    gaus_elim_seq(a_sequential, a0_sequential, n);
    gettimeofday(&end_time, 0);
    // print the running time
    elapsed_seq = get_seconds(start_time, end_time);
    printf("sequential calculation time: %f\n\n", elapsed_seq);

    double **a_loop_unroll = copy_matrix(a, n, n);
    double *a0_loop_unroll = copy_vector(a0, n * n);
    printf("Starting loop unrolling computation...\n\n");
    gettimeofday(&start_time, 0);
    gaus_elim_loop_unroll(a_loop_unroll, a0_loop_unroll, n);
    gettimeofday(&end_time, 0);
    // print the running time
    elapsed_loop_unroll = get_seconds(start_time, end_time);
    printf("loop unrolling calculation time: %f\n\n", elapsed_loop_unroll);


    // add code
    // double elapsed_mpi, elapsed_mpi_unroll;
    // double **a_mpi = copy_matrix(a, n, n);
    // double *a0_mpi = copy_vector(a0, n*n);
    // printf("Starting mpi computation...\n");
    // gettimeofday(&start_time, 0);
    // mpi_ge(a_mpi, a0_mpi, n);
    // gettimeofday(&end_time, 0);
    // elapsed_mpi = get_seconds(start_time, end_time);
    // printf("mpi calculation time: %f\n\n", elapsed_mpi);
    // if (is_matrix_equal(a_sequential, a_mpi, n)) {
    //     printf("success !\n");
    // }

    // end

    if (is_matrix_equal(a_sequential, a_loop_unroll, n))
    {
        printf("The results of sequential and performance computation are the same!\n");
        printf("The improvement of loop unrolling is %.2f%%\n", 100 * (elapsed_seq - elapsed_loop_unroll) / elapsed_seq);
    }
    else
    {
        printf("The results of sequential and performance computation are different!\n");
        if (n <= 20)
        {
            print_matrix(a_sequential, n, n);
            print_matrix(a_loop_unroll, n, n);
        }
    }

    printf("after GE\n");
    print_matrix(a_sequential, n, n);
    print_matrix(a_loop_unroll, n, n);
}

void print_matrix(double **T, int rows, int cols)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            printf("%.2f\t", T[i][j]);
        }
        printf("\n");
    }
    printf("\n\n");
    return;
}

double **copy_matrix(double **T, int rows, int cols)
{
    double **copy;
    copy = (double **)malloc(rows * sizeof(double *));
    for (int i = 0; i < rows; i++)
    {
        copy[i] = (double *)malloc(cols * sizeof(double));
        for (int j = 0; j < cols; j++)
        {
            copy[i][j] = T[i][j];
        }
    }
    return copy;
}

double *copy_vector(double *T, int size)
{
    double *copy;
    copy = (double *)malloc(size * sizeof(double));
    for (int i = 0; i < size; i++)
    {
        copy[i] = T[i];
    }
    return copy;
}

void gaus_elim_seq(double **a, double *a_0, int n)
{
    int i, j, k;
    int indk;
    double c, amax;
    for (i = 0; i < n - 1; i++)
    {
        // find and record k where |a(k,i)|=max|a(j,i)|
        amax = a[i][i];
        indk = i;
        for (k = i + 1; k < n; k++)
        {
            if (fabs(a[k][i]) > fabs(amax))
            {
                amax = a[k][i];
                indk = k;
            }
        }

        // exit with a warning that a is singular
        if (amax == 0)
        {
            printf("matrix is singular!\n");
            exit(1);
        }
        else if (indk != i) // swap row i and row k
        {
            for (j = 0; j < n; j++)
            {
                c = a[i][j];
                a[i][j] = a[indk][j];
                a[indk][j] = c;
            }
        }

        // store multiplier in place of A(k,i)
        for (k = i + 1; k < n; k++)
        {
            a[k][i] = a[k][i] / a[i][i];
        }

        // subtract multiple of row a(i,:) to zero out a(j,i)
        for (k = i + 1; k < n; k++)
        {
            c = a[k][i];
            for (j = i + 1; j < n; j++)
            {
                a[k][j] -= c * a[i][j];
            }
        }
    }
}

void gaus_elim_loop_unroll(double **a, double *a_0, int n)
{
    int i, j, k;
    int indk;
    double c, amax;
    int repeat, left;
    // struct timeval bk0, bk1, bk2, bk3;

    // double finding_max_time = 0;
    // double swapping_time = 0;
    // double subtract_time = 0;
    for (i = 0; i < n - 1; i++)
    {
        // find and record k where |a(k,i)|=max|a(j,i)|
        // gettimeofday(&bk0, 0);
        amax = a[i][i];
        indk = i;
        for (k = i + 1; k < n; k++)
        {
            if (fabs(a[k][i]) > fabs(amax))
            {
                amax = a[k][i];
                indk = k;
            }
        }
        // gettimeofday(&bk1, 0);

        // exit with a warning that a is singular
        if (amax == 0)
        {
            printf("matrix is singular!\n");
            exit(1);
        }
        else if (indk != i) // swap row i and row k
        {
            for (j = 0; j < n; j++)
            {
                c = a[i][j];
                a[i][j] = a[indk][j];
                a[indk][j] = c;
            }
        }
        // gettimeofday(&bk2, 0);

        double a_ii, a_i0, a_i1, a_i2, a_i3;
        double *row_i = a[i];
        a_ii = row_i[i];

        repeat = (n - i - 1) / 4;
        left = (n - i - 1) % 4;
        double c0, c1, c2, c3, a_ij;
        // subtract multiple of row a(i,:) to zero out a(j,i)
        k = i + 1;
        while (repeat--)
        {
            c0 = a[k][i] / a_ii;
            c1 = a[k + 1][i] / a_ii;
            c2 = a[k + 2][i] / a_ii;
            c3 = a[k + 3][i] / a_ii;
            a[k][i] = c0;
            a[k + 1][i] = c1;
            a[k + 2][i] = c2;
            a[k + 3][i] = c3;

            for (j = i + 1; j < n; j++)
            {
                a_ij = row_i[j];
                a[k][j] -= c0 * a_ij;
                a[k + 1][j] -= c1 * a_ij;
                a[k + 2][j] -= c2 * a_ij;
                a[k + 3][j] -= c3 * a_ij;
            }
            k += 4;
        }
        while (left--)
        {
            c = a[k][i] / a_ii;
            for (j = i + 1; j < n; j++)
            {
                a[k][j] -= c * row_i[j];
            }
            a[k][i] = c;
            k++;
        }
        // gettimeofday(&bk3, 0);
        // finding_max_time += get_seconds(bk0, bk1);
        // swapping_time += get_seconds(bk1, bk2);
        // subtract_time += get_seconds(bk2, bk3);
    }
    // printf("finding_max_time: %f\n", finding_max_time);
    // printf("swapping_time: %f\n", swapping_time);
    // printf("subtract_time: %f\n", subtract_time);
}

bool is_matrix_equal(double **a, double **b, int n)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if (fabs(a[i][j] - b[i][j]) > EPS)
            {
                return false;
            }
        }
    }
    return true;
}

double get_seconds(struct timeval start, struct timeval end)
{
    return (double)(end.tv_sec - start.tv_sec) + (double)(end.tv_usec - start.tv_usec) * 1.0e-6;
}