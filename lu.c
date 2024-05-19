#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

void print_matrix(double **a, int rows, int cols);

double **alloc_mat(int r, int c) {
  double **res = (double **)malloc(sizeof(double *) * r);
  for (int i = 0; i < r; i++) {
    res[i] = (double *)malloc(sizeof(double) * c);
  }
  return res;
}

void swap(double **a, int l, int r, int cols) {
    for (int i = 0; i < cols; i++) {
        double tmp = a[l][i];
        a[l][i] = a[r][i];
        a[r][i] = tmp;
    }
}

void copy_row(double *src, double *dst, int n) {
    for (int i = 0; i < n; i++) {
        dst[i] = src[i];
    }
}

int mpi_ge(int argc, char **argv, double **a, int n) {
    int i, j, k;
    // current rank, num of processors, num of handled rows
    int myrank, nproc, m;
    double **cur_a;
    double *pivot_row;
    MPI_Init(&argc, (char ***)&argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);


    m = n / nproc;
    if (n % nproc != 0)
      m++;
    // allocate space for current sub-matrix
    cur_a = alloc_mat(m, n);
    // allocate space for the pivot row
    pivot_row = (double *)malloc(sizeof(double) * n);
    
    int des, tag;
    MPI_Status status;
    if (myrank==0) {
      for (i = 0; i < m; i++)
        for (j = 0; j < n; j++)
          cur_a[i][j] = a[i * nproc][j];
      for (i = 0; i < n; i++)
        if ((i % nproc) != 0) {
          des = i % nproc;
          tag = i / nproc + 1;
          MPI_Send(&a[i][0], n, MPI_DOUBLE, des, tag, MPI_COMM_WORLD);
        }
    } else {
      for (i = 0; i < m; i++) {
        if (myrank + i * nproc >= n) {
          break;
        }
        MPI_Recv(&cur_a[i][0], n, MPI_DOUBLE, 0, i + 1, MPI_COMM_WORLD,
                 &status);
      }
    }

    int pivot, w;
    for (i = 0; i < m; i++) {
        for (j = 0; j < nproc; j++) {
            pivot = i * nproc + j;
            if (pivot >= n) {
                break;
            }
            int mxid = pivot;
            int needSwap = 0;
            if (myrank == 0) {
                for (k = pivot + 1; k < n; k++) {
                    if (fabs(a[k][pivot]) > fabs(a[mxid][pivot])) {
                        mxid = k;
                    }
                }
                swap(a, pivot, mxid, n);

            }
            MPI_Bcast(&mxid, 1, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Bcast(a[mxid], n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Bcast(a[pivot], n, MPI_DOUBLE, 0, MPI_COMM_WORLD);


            if (mxid % nproc == myrank) {
                copy_row(a[mxid], cur_a[mxid / nproc], n);
            }
            if (pivot % nproc == myrank) {
                copy_row(a[pivot], cur_a[i], n);
                copy_row(cur_a[i], pivot_row, n);
            }
            
            MPI_Bcast(pivot_row, n, MPI_DOUBLE, j, MPI_COMM_WORLD);

            
            // transform row [pivot + 1, (i+1)*nproc)
            k = i;
            int cur_row = k * nproc + myrank, recv_row;
            if (myrank > j && cur_row < n) {
                cur_a[k][pivot] = cur_a[k][pivot] / pivot_row[pivot];
                for (w = pivot + 1; w < n; w++)
                  cur_a[k][w] = cur_a[k][w] - pivot_row[w] * cur_a[k][pivot];

                MPI_Send(cur_a[k], n, MPI_DOUBLE, 0, cur_row, MPI_COMM_WORLD);
                // printf("send %d\n", cur_row);
            }
            if (myrank == 0) {
                for (int p = j+1; p < nproc; p++) {
                    // printf("recv %d\n", k*nproc+p);
                    recv_row = k * nproc + p;
                    if (recv_row >= n) break;
                    MPI_Recv(a[recv_row], n, MPI_DOUBLE, p, recv_row, MPI_COMM_WORLD, &status);
                    // printf("recv %d\n", recv_row);
                }
            }
            
            // transform row [(i+1)*nproc, n)
            for (k = i + 1; k < m; k++) {
                cur_row = k * nproc + myrank;
                if (cur_row >= n) break;
                cur_a[k][pivot] = cur_a[k][pivot] / pivot_row[pivot];
                for (w = pivot + 1; w < n; w++)
                  cur_a[k][w] = cur_a[k][w] - pivot_row[w] * cur_a[k][pivot];

                if (myrank != 0) {
                    MPI_Send(cur_a[k], n, MPI_DOUBLE, 0, cur_row, MPI_COMM_WORLD);
                    // printf("send %d\n", cur_row);
                } else {
                    copy_row(cur_a[k], a[k*nproc], n);
                    for (int p = 1; p < nproc; p++) {
                        recv_row = k * nproc + p;
                        // printf("recv %d\n", recv_row);
                        if (recv_row >= n) break;
                        MPI_Recv(a[recv_row], n, MPI_DOUBLE, p, recv_row, MPI_COMM_WORLD, &status);
                    }
                }
            }

            // if (myrank == 0) {
            //     printf("### after %d\n", pivot);
            //     print_matrix(a, n, n);
            // }
        }


    }
    MPI_Finalize();
    return(0);
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

int main(int argc, char **argv) {
    double **a; // 2D matrix for sequential computation
    int i, j, n;

    n = 1000;
    // scanf("%d", &n);

    printf("Creating and initializing matrices...\n\n");
    a = alloc_mat(n, n);
    srand(time(0));
    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++)
            a[i][j] = (double)rand() / RAND_MAX;
    // a[0][0] = 2;
    // a[0][1] = 1;
    // a[0][2] = 2;
    // a[1][0] = 0.5;
    // a[1][1] = 1.5;
    // a[1][2] = 2;
    // a[2][0] = 2;
    // a[2][1] = -0.67;
    // a[2][2] = -0.67;
    printf("origin matrix\n");
    // print_matrix(a, n, n);
    mpi_ge(argc, argv, a, n);
    printf("transformed matrix\n");
    // print_matrix(a, n, n);
}